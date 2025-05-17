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
from datasets import load_dataset, Dataset
import numpy as np
import pandas as pd

import sys
import os
from copy import deepcopy
from datetime import datetime


from custom_MATH_reward import compute_score, remove_boxed, last_boxed_only_string
from math_verify import verify, parse

from execution import check_correctness


def build_prompt(problem: str) -> str:
    return SYSTEM_PROMPT.format(prompt=problem)

class HumanEvalLoader():
    def __init__(self, dataset, trials, simple_prompt=False):
        data = load_dataset(dataset, trust_remote_code=True, split="test")
        
        test_df = data


        self.trials = trials
        self.simple_prompt = simple_prompt
        
        global SYSTEM_PROMPT

        SYSTEM_PROMPT="""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> final code answer inside </answer>. User: Complete the function based on the given docstring. You must put your final answer (full function with all necessary imports) inside <answer> ```python (code) ```</answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted from within the <answer>```python (code)``` </answer> tags and executed.
            {prompt}
            Assistant: <think>"""

        if self.simple_prompt: SYSTEM_PROMPT = """{prompt}"""
        self.df = test_df.map(lambda x: {"prompt": build_prompt(x['prompt'])})

    @staticmethod
    def batch_inference(llm, sampling_params, prompts):
        
        start = time.time()
        outputs = llm.generate(prompts, sampling_params)
        print(str(len(prompts)) + "size batch costing time: " + str(time.time() - start))
        response_batch = []

        all_text = []
        for output in outputs:
            all_text.extend([obj.text for obj in output.outputs])

        for generated_text in all_text:
            response_batch.append(generated_text)
        return response_batch
    
    def save_res(self, res, tokenizer):
        def extract_code(text):
            def extractor(text):
                pattern = r"```python(.+?)```"
                match = re.findall(pattern, text, re.DOTALL)
                
                if match: 
                    return match[-1].strip()
                    
                pattern = r"```(.+?)```"
                match = re.findall(pattern, text, re.DOTALL)
                
                if match:
                    return match[-1].strip()
                    
            pattern = r"<answer>(.+?)</answer>"
            match = re.search(pattern, text, re.DOTALL)
            if match:
                output = extractor(match.group(1).strip())
                if output is None: return match.group(1).strip()
            else:
                output = extractor(text)
                if output is None: return text.strip()
            
            return output

        accu, corr, wrong = 0.0, 0.0, 0.0
        res_length = 0.0

        for each in res:
            res_length += len(tokenizer.encode(each['model_outputs']))

            if self.simple_prompt:
                result = check_correctness(each, each['prompt'] + each['model_outputs'], timeout=30)
            else: result = check_correctness(each, extract_code(each['model_outputs']), timeout=30)
            
            if result['passed']:
                corr += 1
                continue
            wrong += 1
            
        accu = corr / (corr + wrong)
        avg_res = res_length / len(res)

        return accu, corr, wrong, avg_res

    def evaluate(self, llm, sampling_params, tokenizer):
        response_batch = self.batch_inference(llm, sampling_params, self.test_df['prompt'])
        
        res = []
        
        dataset_copy = []
        for i in range(len(self.test_df)):
            for j in range(self.trials):
                curr = deepcopy(self.test_df[i])
                dataset_copy.append(curr)
        dataset = dataset_copy

        for j in range(0, len(dataset), self.trials):
            curr = dataset[j]
            for trial in range(self.trials):
                curr_copy = deepcopy(curr)
                curr_copy["model_outputs"] = response_batch[j+trial]

                res.append(curr_copy)
         
        # for j, curr in enumerate(dataset):
        #     curr["pred"] = pred_batch[j]
        #     curr["model_outputs"] = response_batch[j]
        #     res.append(curr)
       
        results = {}
        tokens = {}
        for i in range(self.trials):
            batch_data = []
            for j in range(i, len(res), self.trials):
                batch_data.append(res[j])
                    
            accu, corr, wrong, avg_res = self.save_res(batch_data, tokenizer)
            
            results[f"batch{i}_acc"] = accu
            tokens[f"batch{i}_tok"] = avg_res

            print("this batch accu is: {}, corr: {}, wrong: {}, tokens: {}\n".format(str(accu), str(corr), str(wrong), str(avg_res)))
        
        mean_acc, stdev_acc = list(results.values())[0], None
        mean_res, stdev_res = list(tokens.values())[0], None
        if self.trials > 1:
            mean_acc =statistics.mean(results.values())
            stdev_acc = statistics.stdev(results.values())
        
            mean_res =statistics.mean(tokens.values())
            stdev_res = statistics.stdev(tokens.values())
        return mean_acc, stdev_acc, mean_res, stdev_res

    def start_evaluation(self, model, llm, sampling_params, tokenizer, subjects, train_size, args, save_to_file, get_default_dict): 
        ds = self.df.train_test_split(train_size=train_size, shuffle=True, seed=42)
        print(ds)
        self.test_df = ds['test']

        mean, stdev, tokmean, tokstd = self.evaluate(llm, sampling_params, tokenizer) 
        print("Results on HumanEval")
        print("=========")
        print(mean, stdev)
        print(tokmean, tokstd)
        
        result_dict = get_default_dict(model=model, subject=subjects[0], train_size=train_size,
                         mean_acc=mean,
                         stdev=stdev,
                            mean_tok=tokmean,
                            stdev_tok=tokstd)
        save_to_file(result_dict)
