import os
import torch
import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-name-or-path", type=str)
    parser.add_argument("--adapter-path", type=str)
    parser.add_argument("--output-dir", type=str)
    
    return parser.parse_args()

def main(args):
        
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model_name_or_path, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base_model, args.adapter_path).cuda()
    model = model.merge_and_unload()
    model._hf_peft_config_loaded = False
    model.save_pretrained(args.output_dir)
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    
    args = parse_args()
    main(args)