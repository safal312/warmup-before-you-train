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


class MMLUProLoader():
    def __init__(self, dataset, trials):
        def preprocess(test_df):
            res_df = []
            for each in test_df:
                options = []
                for opt in each["options"]:
                    if opt == "N/A":
                        continue
                    options.append(opt)
                each["options"] = options
                res_df.append(each)
            
            return res_df
        
        self.dataset = load_dataset(dataset)
        self.choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]
        self.test_df = preprocess(self.dataset["test"])
        
        subjects = []
        for each in self.test_df:
            if each["category"] not in subjects:
                subjects.append(each["category"])
        self.subjects = sorted(subjects)
        
        self.trials = trials


    def select_by_category(self, subject, samplesize=None):
        df = self.test_df

        res = []
        for each in df:
            if each["category"] == subject:
                res.append(each)

        if samplesize is None: return res
 
        if samplesize < 1:
            samplesize = int(len(res) * samplesize)
        samplesize = int(samplesize)
        sampled_items = random.sample(res, min(len(res), samplesize))
        
        return [item for item in res if item not in sampled_items]



    def generate_cot_prompt(self, curr):
        def format_cot_example(example):
            options = ""
            for i, opt in enumerate(example['options']):
                options += "{}. {}\n".format(self.choices[i], opt)

            prompt = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. 
                User: The following is a multiple choice question about {example['category']}. Think step by step and then finish your answer with <answer> \\boxed{{X}} </answer> where X is the correct letter choice.
                {example['question']}
                Options:
                {options}
                Assistant: <think>"""

            # prompt = f"""{example['question']}
            # Options: {options} The final answer should be in \\boxed{{X}} where X is the correct letter choice.
            # """
            return prompt
        
        final_prompt = ""

        subject = curr["category"]
        final_prompt += format_cot_example(curr)
        return final_prompt
    

    
    @staticmethod
    def batch_inference(llm, sampling_params, inference_batch):
        def extract_answer(text):
            #pattern = r"answer is \(?([A-J])\)?"
            #pattern = r"\\boxed\{([ABCDEFGHIJKL])\}"
            pattern = r"\\boxed\{(?:\\text\{)?\s*([A-J])\s*(?:\})?\}" 
            match = re.search(pattern, text)
            if match:
                return match.group(1)
            else:
                print("1st answer extract failed\n" + text)
                return extract_again(text)


        def extract_again(text):
            pattern = r"<answer>\s*([A-J])(\.\s*(.*?))?\s*</answer>"
            match = re.search(pattern, text)
            if match:
                return match.group(1)
            else:
                return extract_final(text)


        def extract_final(text):
            pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(0)
            else:
                return None
        
        start = time.time()
        outputs = llm.generate(inference_batch, sampling_params)
        print(str(len(inference_batch)) + "size batch costing time: " + str(time.time() - start))
        response_batch = []
        pred_batch = []

        all_text = []
        for output in outputs:
            all_text.extend([obj.text for obj in output.outputs])

        for generated_text in all_text:
            response_batch.append(generated_text)
            pred = extract_answer(generated_text)
            pred_batch.append(pred)
        return pred_batch, response_batch
    
    @staticmethod
    def save_res(res, tokenizer):
        accu, corr, wrong = 0.0, 0.0, 0.0
        res_length = 0.0

        for each in res:
            res_length += len(tokenizer.encode(each['model_outputs']))
            if not each["pred"]:
                x = random.randint(0, len(each["options"]) - 1)
                if x == each["answer_index"]:
                    corr += 1
                    # print("random hit.")
                else:
                    wrong += 1
            elif each["pred"] == each["answer"]:
                corr += 1
            else:
                wrong += 1
        if corr + wrong == 0:
            return 0.0, 0.0, 0.0
        accu = corr / (corr + wrong)
        avg_res = res_length / len(res)

        return accu, corr, wrong, avg_res

    def evaluate(self, dataset, llm, sampling_params, tokenizer):
        inference_batch = []
        for i in tqdm(range(len(dataset))):
            curr = dataset[i]
            prompt = self.generate_cot_prompt(curr)
        
            inference_batch.append(prompt)
             
        pred_batch, response_batch = self.batch_inference(llm, sampling_params, inference_batch)
        
        res = []
    
        dataset_copy = []
        for i in range(len(dataset)):
            for j in range(self.trials):
                curr = deepcopy(dataset[i])
                dataset_copy.append(curr)
        dataset = dataset_copy

        for j in range(0, len(dataset), self.trials):
            curr = dataset[j]
            for trial in range(self.trials):
                curr_copy = deepcopy(curr)
                curr_copy["pred"] = pred_batch[j+trial]
                curr_copy["model_outputs"] = response_batch[j+trial]

                res.append(curr_copy)
         
        # for j, curr in enumerate(dataset):
        #     curr["pred"] = pred_batch[j]
        #     curr["model_outputs"] = response_batch[j]
        #     res.append(curr)
        print(len(dataset))
        print(len(res))
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
        return mean_acc, stdev_acc, mean_res, stdev_res, results, tokens, res

    def start_evaluation(self, model, llm, sampling_params, tokenizer, subjects, train_size, args, save_to_file, get_default_dict):
        subject_res = []
        subject_toks = []
        all_results_acc = {}
        all_results_tok = {}
        all_res = []
        for subject in subjects:       
            sub_ds = self.select_by_category(subject, train_size) 
            mean, stdev, tokmean, tokstd, results_dict, tokens_dict, responses = self.evaluate(sub_ds, llm, sampling_params, tokenizer) 
            print(subject)
            print("=========")
            print(mean, stdev)
            print(tokmean, tokstd)

            result_dict = get_default_dict(model=model, subject=subject, train_size=train_size,
                             mean_acc=mean,
                             stdev=stdev,
                            mean_tok=tokmean,
                            stdev_tok=tokstd)
            subject_res.append(mean)
            subject_toks.append(tokmean)
            save_to_file(result_dict)
            all_res.extend(responses)
            all_results_acc[subject] = results_dict.values()
            all_results_tok[subject] = tokens_dict.values()
        
        df = pd.DataFrame(all_res)
        fname = "--".join([model.replace(".", "").replace("/", "--"), "|".join(subjects), str(train_size)])
        if not os.path.exists("files"): os.mkdir("./files")
        df.to_csv(f"files/{fname}.csv", index=False)

        all_accs = zip(*all_results_acc.values())
        all_toks = zip(*all_results_tok.values())
        macro_accs = []
        macro_toks = []
        for acc in all_accs:
            acc = list(acc)
            macro_accs.append(statistics.mean(acc))
        
        for tok in all_toks:
            tok = list(tok)
            macro_toks.append(statistics.mean(tok))

        if len(subjects) > 1:
            result_dict = get_default_dict(model=model, subject="all", train_size=train_size,
                             mean_acc=statistics.mean(macro_accs),
                             stdev=statistics.stdev(macro_accs),
                            mean_tok=statistics.mean(macro_toks),
                            stdev_tok=statistics.stdev(macro_toks))
            save_to_file(result_dict)
