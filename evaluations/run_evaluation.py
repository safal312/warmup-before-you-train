import statistics
import csv
import json
import argparse
import random
import transformers
import torch
import time
import re
from vllm import LLM, SamplingParams
from tqdm import tqdm
#import logging
import sys
from datasets import load_dataset
import numpy as np
import pandas as pd

import sys
import os
from copy import deepcopy
from datetime import datetime

from mmlu_loader import MMLUProLoader
from math_loader import MATH500Loader
from humaneval_loader import HumanEvalLoader

import vllm
from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

import wandb


random.seed(42)
max_model_length = 4096
max_new_tokens = 8192
out_dir = "all_outputs"
# output_file = None

# out_df = None

def load_model(args, model):
    
    sampling_params = SamplingParams(
            n = args.trials,
            temperature=0.7,  #0.7, 
                                     top_p=0.95, #0.95,
                                     max_tokens=max_new_tokens,
                                        stop=["Question:",
                                              "Assistant:",
                                              "User:"])

    llm = LLM(model=model, gpu_memory_utilization=0.9,
            tensor_parallel_size=torch.cuda.device_count(),
            max_model_len=max_model_length,
            trust_remote_code=True)
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    return llm, sampling_params, tokenizer

def get_default_dict(**kwargs):
    now = datetime.now()
    details = {
            'timestamp': now.strftime("%Y-%m-%d_%H-%M-%S"),
            'model': kwargs['model'],
            'subject': kwargs['subject'],
            'train_size': kwargs['train_size'],
            'mean_acc': kwargs['mean_acc'],
            'stdev': kwargs['stdev'],
            'mean_tok': kwargs['mean_tok'],
            'stdev_tok': kwargs['stdev_tok']
            }

    wandb.log(details)
    return details

def save_to_file(row_dict):
    global out_df
    temp_df = pd.DataFrame(row_dict, index=[0])
    out_df = pd.concat([out_df, temp_df])

    # Identify float columns
    float_cols = out_df.select_dtypes(include=['float64', 'float32']).columns

    # Round the values in the float columns to 3 decimal place
    out_df[float_cols] = out_df[float_cols].round(3)
    out_df.to_csv(output_file, index=False)

def generate_and_save(model, subjects, train_size, ds_mmlu, ds_math, ds_humeval, args):
    llm, sampling_params, tokenizer = load_model(args, model)
    if subjects[0] != "human_eval" and subjects[0] != "math500": ds_mmlu.start_evaluation(model, llm, sampling_params, tokenizer, subjects, train_size, args, save_to_file, get_default_dict)
    else:
        ds_math.start_evaluation(model, llm, sampling_params, tokenizer, ["math500"], 1, args, save_to_file, get_default_dict)
        ds_humeval.start_evaluation(model, llm, sampling_params, tokenizer, ["human_eval"], train_size, args, save_to_file, get_default_dict)
    cleanup_dist_env_and_memory()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple script to process terminal inputs.")
    parser.add_argument("--input", help="Path to the input CSV file")
    parser.add_argument("--run", help="The command to run")
    parser.add_argument("--out_file", help="The output file")
    parser.add_argument("--trials", type=int, help="Sample size")

    args = parser.parse_args()
    
    global out_df, output_file

    output_file = args.out_file

    if os.path.exists(output_file):
        out_df = pd.read_csv(output_file)
    else:
        out_df = pd.DataFrame()

    print("Saving to:", output_file)
    df = pd.read_csv(args.input)
    ds_mmlu = MMLUProLoader("TIGER-Lab/MMLU-Pro", args.trials)
    
    ds_math = MATH500Loader("HuggingFaceH4/MATH-500", args.trials)
    
    ds_humeval = HumanEvalLoader("evalplus/humanevalplus", args.trials, simple_prompt=False)
    
    for index, row in df.iterrows():
        if not row['select']: continue

        models = os.listdir(row['model']) if row['model'].startswith("../") and 'checkpoint-' not in row['model'] else [row['model']]
        wandb.init(project="evaluations", name=f"{row['model']}--{row['subjects']}")
        for index, model in enumerate(models):
            if row['model'].startswith("../") and 'checkpoint-' not in row['model']: model = row['model'] + "/" + model
            
    
            # dataset = ds_mmlu
            subjects = []
            # if row['subjects'] == 'math500':
            #     subjects=['math500']
            #     dataset=ds_math
            if row['subjects'] != 'all':
                subjects = sorted(row['subjects'].split('|'))
            else:
                subjects = ds_mmlu.subjects
            
            if row['train_size'] > 1: row['train_size'] = int(row['train_size'])
            generate_and_save(model, subjects, row['train_size'], ds_mmlu, ds_math, ds_humeval, args)
    
        wandb.finish()


