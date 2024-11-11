import os
import re
import json
import string
import argparse
import numpy as np
from tqdm import tqdm

import torch
import tiktoken
from datasets import load_dataset, load_from_disk, concatenate_datasets, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from openai import OpenAI, AzureOpenAI

from prompts import PROMPTS

def inference_vllm(dataset, model, max_tokens, temperature):
    
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    outputs = model.generate([item["prompt"] for item in dataset], sampling_params)
    
    def process(item, idx):
        item["prediction"] = outputs[idx].outputs[0].text
        return item
    
    dataset = dataset.map(process, with_indices=True, num_proc=8)

    return dataset

def load_data(args, tokenizer):
    
    dataset = concatenate_datasets([
        load_dataset("THUDM/LongBench", "narrativeqa", split="test"),
        load_dataset("THUDM/LongBench", "qasper", split="test"),
        load_dataset("THUDM/LongBench", "multifieldqa_en", split="test"),
        load_dataset("THUDM/LongBench", "hotpotqa", split="test"),
        load_dataset("THUDM/LongBench", "2wikimqa", split="test"),
        load_dataset("THUDM/LongBench", "musique", split="test")
    ])

    def process(item):

        item["prompt"] = tokenizer.apply_chat_template(
            [{"role": "user", "content": PROMPTS["eval_longbench_qa"].format(context=item["context"], input=f"{item['input']}\n{PROMPTS[args.prompt]}".strip())}],
            add_generation_prompt=True,
            tokenize=False
        )
        return item
    
    dataset = dataset.map(process, num_proc=8)
    
    return dataset

def inference(args):
    
    model = LLM(model=args.model_name_or_path, tensor_parallel_size=args.tensor_parallel_size, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    dataset = load_data(args, tokenizer)
    dataset = inference_vllm(dataset, model, max_tokens=args.max_tokens, temperature=args.temperature)
    
    dataset.save_to_disk(args.output_path)
    
def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def substring_exact_match_score(prediciton, ground_truth):
    """Check if the ground truth is a (soft) exact match substring of the prediction."""
    return normalize_answer(ground_truth) in normalize_answer(prediciton) 

def drqa_metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """Given a prediction and multiple valid answers, return the score of
    the best prediction-answer_n pair given a metric function.
    """
    # ground truth could be a string or a list of strings or a list of list of strings
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    elif isinstance(ground_truths[0], list):
        ground_truths = [ground_truth for ground_truths_list in ground_truths for ground_truth in ground_truths_list]

    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def evaluate_sub_em(args):
    
    dataset = load_from_disk(args.dataset)
    print(dataset)

    def process(item):
        item["sub_em"] = drqa_metric_max_over_ground_truths(substring_exact_match_score, item["prediction"], item["answers"])
        return item
    
    dataset = dataset.map(process, num_proc=8)
    
    metrics = {key: [] for key in set(dataset["dataset"])}
    for item in dataset:
        metrics[item["dataset"]].append(item["sub_em"])
    
    for dataset in metrics.keys():    
        metrics[dataset] = {
            "sub_em": np.mean(metrics[dataset]) * 100,
            "num_samples": len(metrics[dataset])
        }
    
    json.dump(metrics, open(args.output_path, "w", encoding="utf-8"), indent=4)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--split", type=str)
    parser.add_argument("--stage", type=str)
    parser.add_argument("--eval_strategy", type=str)
    parser.add_argument("--output_path", type=str)

    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--tensor_parallel_size", type=int)
    parser.add_argument("--max_tokens", type=int)
    parser.add_argument("--temperature", type=float)
    
    parser.add_argument("--openai", type=str)
    
    parser.add_argument("--prompt", type=str)

    args = parser.parse_args()
    
    if args.stage == "inference":
        inference(args)
    elif args.stage == "evaluation":
        if args.eval_strategy == "sub_em":
            evaluate_sub_em(args)
