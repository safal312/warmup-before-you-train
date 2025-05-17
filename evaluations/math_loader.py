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

SYSTEM_PROMPT = """
A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively.
i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>.
And your final answer will be extracted automatically by the \\boxed{{}} tag.
{prompt}
Assistant: <think>
"""

# SYSTEM_PROMPT = """{prompt}
# """

def build_prompt(problem: str) -> str:
    return SYSTEM_PROMPT.format(prompt=problem)

class MATH500Loader():
    def __init__(self, dataset, trials):
        data = load_dataset(dataset, trust_remote_code=True)['test']
        prompts = []
        ground_truths = []
        for example in data:
            prompt = build_prompt(example['problem'])
            gt = last_boxed_only_string(example['solution'])
            prompts.append(prompt)
            ground_truths.append(gt)
        self.test_df = Dataset.from_dict({'prompt': prompts,
                                          'gt': ground_truths})

        self.trials = trials


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
    
    @staticmethod
    def save_res(res, tokenizer):
        accu, corr, wrong = 0.0, 0.0, 0.0
        res_length = 0.0

        for each in res:
            res_length += len(tokenizer.encode(each['model_outputs']))
            if verify(parse(each['model_outputs']), parse(str(each['gt']))):
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
        mean, stdev, tokmean, tokstd = self.evaluate(llm, sampling_params, tokenizer) 
        print("Results on MATH500")
        print("=========")
        print(mean, stdev)
        print(tokmean, tokstd)
        
        result_dict = get_default_dict(model=model, subject=subjects[0], train_size=train_size,
                         mean_acc=mean,
                         stdev=stdev,
                            mean_tok=tokmean,
                            stdev_tok=tokstd)
        save_to_file(result_dict)
