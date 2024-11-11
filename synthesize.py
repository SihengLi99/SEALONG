import os
import re
import json
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

import faiss
import torch
import tiktoken
from openai import OpenAI, AzureOpenAI
from datasets import load_dataset, load_from_disk, Dataset
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from vllm import LLM, SamplingParams
from langchain_text_splitters import RecursiveCharacterTextSplitter

from prompts import PROMPTS
from eval_longbench_qa import drqa_metric_max_over_ground_truths, substring_exact_match_score

def inference_temperature_sampling(args, dataset):

    model = LLM(model=args.model_name_or_path, tensor_parallel_size=args.tensor_parallel_size, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    def process(item):
        item["raw_prompt"] = PROMPTS["eval_longbench_qa"].format(context=item["context"], input=f"{item['input']}\n{PROMPTS[args.prompt]}".strip())
        item["prompt"] = tokenizer.apply_chat_template(
            [{"role": "user", "content": item["raw_prompt"]}],
            add_generation_prompt=True,
            tokenize=False
        )
        return item
    dataset = dataset.map(process, num_proc=8)
    
    num_iter = args.n // args.n_iter
    print(f"N: {args.n} - N_ITER: {args.n_iter} - Num Iteration: {num_iter}")
    total_outputs = []
    for _ in range(num_iter):
        sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens, n=args.n_iter)
        total_outputs.append(model.generate([item["prompt"] for item in dataset], sampling_params))
    
    def process(item, idx):
        item["predictions"] = sum([[output.text for output in outputs[idx].outputs] for outputs in total_outputs], [])
        assert len(item["predictions"]) == args.n
        return item
    dataset = dataset.map(process, num_proc=8, with_indices=True)
    
    dataset.save_to_disk(args.sample_dataset_output_path)
    
    return dataset

def encode_sentences(dataset, encoder_model_name_or_path):
    
    sentences = sum([item["predictions"] for item in dataset], [])
    
    if "jinaai/jina-embeddings-v3" in encoder_model_name_or_path:
        model = AutoModel.from_pretrained(encoder_model_name_or_path, trust_remote_code=True).cuda()
        sentence_embeddings = model.encode(sentences, task="text-matching", max_length=8192).tolist()
    elif "nvidia/NV-Embed-v2" in encoder_model_name_or_path:
        model = SentenceTransformer('nvidia/NV-Embed-v2', trust_remote_code=True).cuda()
        model.max_seq_length = 32768
        model.tokenizer.padding_side="right"
        def add_eos(input_examples):
            input_examples = [input_example + model.tokenizer.eos_token for input_example in input_examples]
            return input_examples
        batch_size = 512
        sentence_embeddings = model.encode(add_eos(sentences), batch_size=batch_size, normalize_embeddings=True)
    elif "mixedbread-ai/mxbai-embed-large-v1" in encoder_model_name_or_path:
        model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1").cuda()
        sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1).tolist()
    elif "openbmb/MiniCPM-Embedding" in encoder_model_name_or_path:
        model = SentenceTransformer(encoder_model_name_or_path, trust_remote_code=True, model_kwargs={"attn_implementation": "flash_attention_2", "torch_dtype": torch.float16})
        sentence_embeddings = model.encode(sentences).tolist()

    return sentence_embeddings
        
def minimum_bayes_risk_score_sentence_embedding(args, dataset):
    
    sentence_embeddings = encode_sentences(dataset, args.encoder_model_name_or_path)

    num_predictions = len(dataset[0]["predictions"])
    assert len(sentence_embeddings) == len(dataset) * num_predictions        
    
    def process(item, idx):
        embeddings = np.array(sentence_embeddings[idx*num_predictions: (idx+1)*num_predictions])
        similarity = embeddings @ embeddings.T
        item["mbr_scores_embedding"] = similarity.mean(1).tolist()
        item["prediction"] = item["predictions"][np.argmax(item["mbr_scores_embedding"])]
        return item
    dataset = dataset.map(process, num_proc=8, with_indices=True)
    
    dataset.save_to_disk(args.score_dataset_output_path)    
    
    return dataset

def synthesize(args):
    
    raw_dataset = load_from_disk(args.raw_dataset)
        
    try:
        sample_dataset = load_from_disk(args.sample_dataset)
    except:
        sample_dataset = inference_temperature_sampling(args, raw_dataset)
    
    try:
        score_dataset = load_from_disk(args.score_dataset)
    except:
        score_dataset = minimum_bayes_risk_score_sentence_embedding(args, sample_dataset)

    print(raw_dataset)
    print(sample_dataset)
    print(score_dataset)
    
    def process(item):

        sorted_index = np.argsort(item["mbr_scores_embedding"])
        
        # item["chosen"] = item["predictions"][sorted_index[-2]]
        item["chosen"] = item["prediction"]
        item["rejected"] = item["predictions"][random.choice(sorted_index[:sorted_index.shape[-1]//2])]
        
        item["sub_ems"] = [drqa_metric_max_over_ground_truths(substring_exact_match_score, prediction, [item["answer"]]) for prediction in item["predictions"]]
        item["sub_em_max"] = max(item["sub_ems"])
        item["sub_em_random"] = random.choice(item["sub_ems"])
        item["sub_em_average"] = np.mean(item["sub_ems"])
        
        item["sub_em_prediction"] = drqa_metric_max_over_ground_truths(substring_exact_match_score, item["prediction"], [item["answer"]])
        item["sub_em_chosen"] = drqa_metric_max_over_ground_truths(substring_exact_match_score, item["chosen"], [item["answer"]])
        item["sub_em_rejected"] = drqa_metric_max_over_ground_truths(substring_exact_match_score, item["rejected"], [item["answer"]])
        
        return item
    score_dataset = score_dataset.map(process, num_proc=8)
    
    metrics = {
        "sub_em_max": np.mean([item["sub_em_max"] for item in score_dataset]) * 100,
        "sub_em_random": np.mean([item["sub_em_random"] for item in score_dataset]) * 100,
        "sub_em_average": np.mean([item["sub_em_average"] for item in score_dataset]) * 100,
        "sub_em_prediction": np.mean([item["sub_em_prediction"] for item in score_dataset]) * 100,
        "sub_em_chosen": np.mean([item["sub_em_chosen"] for item in score_dataset]) * 100,
        "sub_em_rejected": np.mean([item["sub_em_rejected"] for item in score_dataset]) * 100,
    }
    print(json.dumps(metrics, indent=4))
    json.dump(metrics, open(args.output_path, "w"), indent=4)
    
    def process(item):
        return {
            "messages": [
                {"role": "user", "content": item["raw_prompt"]},
                {"role": "assistant", "content": item["prediction"]}
            ],
            "prompt": item["raw_prompt"],
            "chosen": item["chosen"],
            "rejected": item["rejected"]
        }
    dataset = score_dataset.map(process, num_proc=8, remove_columns=[column for column in score_dataset.column_names if column not in ["messages", "prompt", "chosen", "rejected"]])
    print(dataset)
    print(args.dataset_output_path)
    dataset.save_to_disk(args.dataset_output_path)    

            
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str)
    parser.add_argument("--raw_dataset", type=str)
    parser.add_argument("--sample_dataset", type=str)
    parser.add_argument("--score_dataset", type=str)
    
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--dataset_output_path", type=str)
    parser.add_argument("--sample_dataset_output_path", type=str)
    parser.add_argument("--score_dataset_output_path", type=str)
    
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--tensor_parallel_size", type=int)
    parser.add_argument("--max_model_len", type=int)
    parser.add_argument("--gpu_memory_utilization", type=float)
    
    parser.add_argument("--max_tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--n", type=int)
    parser.add_argument("--n_iter", type=int)
    
    parser.add_argument("--encoder_model_name_or_path", type=str)
    
    parser.add_argument("--prompt", type=str)
    
    args = parser.parse_args()
        
    synthesize(args)