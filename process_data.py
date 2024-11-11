import os
import re
import ast
import json
import random
import urllib.parse
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

import tiktoken
from transformers import AutoTokenizer
from datasets import (
    load_dataset,
    load_from_disk,
    Dataset,
    DatasetDict
)

from prompts import PROMPTS

def extract_title(url):
    title = url.split('/')[-1].replace('_', ' ')
    title = urllib.parse.unquote(title)
    title = title.split("#")[0].strip()
    return title

# Function to normalize the strings
def normalize_string(s):
    # Convert to lowercase
    s = s.lower()
    # Remove periods
    s = s.replace('.', '')
    # Standardize hyphens (convert long dashes to short ones)
    s = s.replace('–', '-').replace('—', '-')
    # Replace multiple spaces with a single space and strip leading/trailing spaces
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def process_item(item):
    return item[0], normalize_string(item[1]["title"])
    
def process_musique():
    
    seed = 42
    output_path = f"./data/musique/musique_processed_{seed}"
        
    with open("./data/musique/musique_ans_v1.0_train.jsonl", "r") as f:
        dataset = [json.loads(line) for line in f]
    
    print(dataset[0].keys())

    wikipedia = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
    
    try:
        title2idx = json.load(open("./data/wikipedia_title2idx.json", "r"))
    except:
        title2idx = {}
        with Pool(processes=64) as pool:
            for idx, title in tqdm(pool.imap(process_item, enumerate(wikipedia)), total=len(wikipedia)):
                title2idx[title] = idx
        json.dump(title2idx, open("./data/wikipedia_title2idx.json", "w"))
                
    corpus = []
    for item in tqdm(dataset):
        
        assert item["answerable"] is True
        
        support_docs = set([normalize_string(para["title"]) for para in item["paragraphs"] if para["is_supporting"] is True])
        unsupport_docs = set([normalize_string(para["title"]) for para in item["paragraphs"] if para["is_supporting"] is False and normalize_string(para["title"]) in title2idx])
        
        if any([title not in title2idx for title in support_docs]):
            continue
        
        item["support_docs"] = list(support_docs)
        item["unsupport_docs"] = list(unsupport_docs)
        
        corpus.append(item)

    dataset = Dataset.from_list(corpus).shuffle(seed)
    print(dataset)
    
    dataset.save_to_disk(output_path)

def create_dataset_musique():
    
    seed = 42
    min_tokens = 4
    max_tokens = 31
    num_samples = 2
    output_path = f"./data/musique/musique_processed_{seed}_{min_tokens}k_{max_tokens}k_{num_samples}k"
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    
    dataset = load_from_disk(f"./data/musique/musique_processed_{seed}")
    
    wikipedia = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
    
    try:
        title2idx = json.load(open("./data/wikipedia_title2idx.json", "r"))
    except:
        title2idx = {}
        with Pool(processes=64) as pool:
            for idx, title in tqdm(pool.imap(process_item, enumerate(wikipedia)), total=len(wikipedia)):
                title2idx[title] = idx
        json.dump(title2idx, open("./data/wikipedia_title2idx.json", "w"))
    
    def process(item):
        docs = [title for title in item["support_docs"]]
        
        num_tokens = 0
        for title in docs:
            num_tokens += len(tokenizer.tokenize(wikipedia[title2idx[title]]["text"]))
        
        target_tokens = random.randint(min_tokens, max_tokens) * 1024
        unsupport_docs = random.sample(item["unsupport_docs"], len(item["unsupport_docs"]))
        for title in unsupport_docs:
            text_tokens = len(tokenizer.tokenize(wikipedia[title2idx[title]]["text"]))
            if num_tokens + text_tokens <= target_tokens:
                docs.append(title)
                num_tokens += text_tokens
                
        random.shuffle(docs)

        context = ""
        for idx, title in enumerate(docs):
            text = wikipedia[title2idx[title]]["text"]
            context += f"Passage {idx+1}:\n{text}".strip() + "\n"
        context = context.strip()
        
        return {
            "context": context,
            "input": item["question"],
            "answer": item["answer"],
            "num_tokens": len(tokenizer.tokenize(context))
        }

    dataset = dataset.map(process, num_proc=8, remove_columns=[column for column in dataset.column_names if column not in ["context", "input", "answer", "num_tokens"]])    
    dataset = dataset.select(range(int(num_samples * 1024)))
    print(dataset)
    print(np.mean(dataset["num_tokens"]))
    
    dataset.save_to_disk(output_path)
    
    
def create_dataset_musique_qwen():
    
    seed = 42
    min_tokens = 4
    max_tokens = 31
    num_samples = 2
    output_path = f"./data/musique/musique_processed_{seed}_{min_tokens}k_{max_tokens}k_{num_samples}k_qwen"
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-72B-Instruct")
    
    dataset = load_from_disk(f"./data/musique/musique_processed_{seed}")
    
    wikipedia = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
    
    try:
        title2idx = json.load(open("./data/wikipedia_title2idx.json", "r"))
    except:
        title2idx = {}
        with Pool(processes=64) as pool:
            for idx, title in tqdm(pool.imap(process_item, enumerate(wikipedia)), total=len(wikipedia)):
                title2idx[title] = idx
        json.dump(title2idx, open("./data/wikipedia_title2idx.json", "w"))
    
    def process(item):
        docs = [title for title in item["support_docs"]]
        
        num_tokens = 0
        for title in docs:
            num_tokens += len(tokenizer.tokenize(wikipedia[title2idx[title]]["text"]))
        
        target_tokens = random.randint(min_tokens, max_tokens) * 1024
        unsupport_docs = random.sample(item["unsupport_docs"], len(item["unsupport_docs"]))
        for title in unsupport_docs:
            text_tokens = len(tokenizer.tokenize(wikipedia[title2idx[title]]["text"]))
            if num_tokens + text_tokens <= target_tokens:
                docs.append(title)
                num_tokens += text_tokens
                
        random.shuffle(docs)

        context = ""
        for idx, title in enumerate(docs):
            text = wikipedia[title2idx[title]]["text"]
            context += f"Passage {idx+1}:\n{text}".strip() + "\n"
        context = context.strip()
        
        return {
            "context": context,
            "input": item["question"],
            "answer": item["answer"],
            "num_tokens": len(tokenizer.tokenize(context))
        }

    dataset = dataset.map(process, num_proc=8, remove_columns=[column for column in dataset.column_names if column not in ["context", "input", "answer", "num_tokens"]])   
    print(dataset)
    dataset = dataset.filter(
        lambda item: item["num_tokens"] <= max_tokens * 1024,
        num_proc=8
    ) 
    print(dataset)   
 
    dataset = dataset.select(range(int(num_samples * 1024)))
    print(dataset)
    print(np.mean(dataset["num_tokens"]))
    
    dataset.save_to_disk(output_path)
    
if __name__ == "__main__":
                
    process_musique()
    
    create_dataset_musique()

    create_dataset_musique_qwen()
