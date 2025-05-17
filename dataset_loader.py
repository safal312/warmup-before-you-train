import os
import re
import json
import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset, concatenate_datasets

from custom_MATH_reward import compute_score, remove_boxed, last_boxed_only_string

from execution import check_correctness

def extract_boxed_answer(solution):
    return last_boxed_only_string(solution)


    

def load_math(sample=None, diff_filter=False):
    """Helper function for MATH to load train, test, and reward function"""
    SYSTEM="""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> final answer inside \\boxed{{}} tag </answer>. User: You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted automatically by the \\boxed{{}} tag.
    {prompt}
    Assistant: <think>"""

    def reward_correct(completions, answer, **kwargs):
        # check if the strings ends with </think><answer>[boxed answer]</answer>
        def check_format(s, gt):
            pattern = r".+</think>\s*<answer>(.+)</answer>\s*$"
            if not (s.count("</think>") == 1 and s.count("<answer>") == 1 and s.count("</answer>") == 1):
                # incorrect amount of tokens
                return -2 
            match = re.search(pattern, s, re.DOTALL)
            # if answer doesn't match provided format
            if not match: return -2

            # answer format is correct now
            # look for boxed tag
            ext_string = last_boxed_only_string(match.group(1))
            if ext_string is None: return -1   #No boxed tag found
            # ext_string = match.group(1)
            
            # if correct, then reward 2
            if verify(parse(ext_string), parse(gt)): return 2
            else: return -0.5 # extracted but incorrect then reward -0.5

        return [check_format(c, gt) for c, gt in zip(completions, answer)]

    train = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="train")
    test = load_dataset("HuggingFaceH4/MATH-500", split="test")

    #sample 'n' from train
    if sample is not None:
        train = train.train_test_split(train_size=sample, shuffle=True, seed=42)['train']

    train = train.map(lambda x: {
        "prompt": SYSTEM.format(prompt=x["problem"]),
        "answer": extract_boxed_answer(x["solution"]),
        "level": x["level"]
        })

    test = test.map(lambda x: {
        "prompt": SYSTEM.format(prompt=x["problem"]),
        "answer": x["answer"],
        "level": x["level"]
        })

    if diff_filter:
        train = train.filter(lambda x: (x['level'] == 'Level 4') or (x['level'] == 'Level 5'))
    
    train = train.remove_columns(["problem", "solution", "type"])
    test = test.remove_columns(["problem", "solution", "subject", "unique_id"])
    return train, test, reward_fn

def load_mmlupro(sample=None, categories=None):
    """Helper function for MMLU-Pro to load train, test, and reward function.
    Extraction mechanism is built from https://github.com/TIGER-AI-Lab/MMLU-Pro"""
    def reward_mmlu(completions, answer, **kwargs):
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

        def calculate_reward(output, ground_truth):
            def preprocess(text):
                return extract_answer(text)

            pattern = r".+</think>\s*<answer>(.+)</answer>\s*$"
            s = output 
            if not (s.count("</think>") == 1 and s.count("<answer>") == 1 and s.count("</answer>") == 1):
                # incorrect amount of tokens
                #print("counts no match")
                return -2 
            match = re.search(pattern, s, re.DOTALL)
            # if answer doesn't match provided format
            if not match: return -2

            # answer format is correct now
            # 
            
            gt = ground_truth
            out = preprocess(match.group(1).strip())

            # formatting of final answer also good
            return 2 if gt == out else -0.5
        
        return [calculate_reward(c, gt) for c, gt in zip(completions, answer)]
    
    
    def clean_options(example):
        example["options"] = [opt for opt in example["options"] if opt != "N/A"]
        return example
    
    def create_train_test_split_by_category(dataset, samples_per_category=50, seed=42):
        df = dataset.to_pandas()
        
        if categories is None:
            all_categories = df['category'].unique()
        else:
            all_categories = categories

        train_dfs = []
        test_dfs = []

        rng = np.random.default_rng(seed)
        
        percent = False
        if (samples_per_category) < 1:
            percent = True
        
        sample_per = samples_per_category

        for category in all_categories:
            samples_per_category = sample_per
            category_df = df.loc[df['category'] == category]
            
            # percent based sampling
            if percent:
                samples_per_category = int(len(category_df) * samples_per_category)
            
            print(category, samples_per_category)

            if len(category_df) >= samples_per_category:
                sampled_indices = rng.choice(category_df.index, size=samples_per_category, replace=False)
                train_category_df = category_df.loc[sampled_indices]
                test_category_df = category_df.drop(sampled_indices)
                train_dfs.append(train_category_df)
                test_dfs.append(test_category_df)
            else:
                print(f"Warning: Category '{category}' has fewer than {samples_per_category} samples ({len(category_df)}). All samples from this category will be included in the training set and the test set will have no samples from this category.")
                train_dfs.append(category_df)
                test_dfs.append(pd.DataFrame()) # Append an empty DataFrame for the test set

        train_df = pd.concat(train_dfs).reset_index(drop=True)
        test_df = pd.concat(test_dfs).reset_index(drop=True)

        train_dataset = Dataset.from_pandas(train_df)
        test_dataset = Dataset.from_pandas(test_df)
        print("Train dataset size:", len(train_dataset))
        print("Test dataset size:", len(test_dataset))

        return train_dataset, test_dataset, reward_mmlu

    dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    dataset = dataset.map(clean_options)

    trainds, testds, reward_mmlu = create_train_test_split_by_category(dataset, sample)
    
    def apply_prompt(example):
        choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]
        options = ""
        for i, opt in enumerate(example['options']):
            options += "{}. {}\n".format(choices[i], opt)
        
        SYSTEM_PROMPT = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. 
            User: The following are multiple choice questions (with answers) about {example['category']}. Think step by step and then finish your answer with <answer> \\boxed{{X}} </answer> where X is the correct letter choice.
            {example['question']}
            Options:
            {options}
            Assistant: <think>"""

        return {"prompt": SYSTEM_PROMPT, "answer": example["answer"]}

    
    trainds = trainds.map(apply_prompt)
    testds = testds.map(apply_prompt)
    return trainds, testds, reward_mmlu


def load_kk():  
    """Helper function to load train, test dataset, and reward function for knights and knaves dataset"""
    def reward_kk(completions, answer, **kwargs):
        def calculate_reward(output, ground_truth):
            def preprocess(text):
               # pattern = r".+</think>\s*<answer>(.+)</answer>\s*$"
               # match = re.search(pattern, text, re.DOTALL)
               # if match: text = match.group(1)

                lines = text.split("\n")
                solution = {}

                for line in lines:
                    sp = line.strip().split(" ")
                    if not line: continue
                    if len(sp) < 2: continue

                    solution[sp[1].lower()] = sp[-1].lower()
                return solution 

            pattern = r".+</think>\s*<answer>(.+)</answer>\s*$"
            s = output 
            if not (s.count("</think>") == 1 and s.count("<answer>") == 1 and s.count("</answer>") == 1):
                # incorrect amount of tokens
                #print("counts no match")
                return -2 
            match = re.search(pattern, s, re.DOTALL)
            # if answer doesn't match provided format
            if not match: return -2

            # answer format is correct now
            # 
            
            gt = preprocess(ground_truth)
            out = preprocess(match.group(1).strip())
            names = out.keys()
            for name in names:
                val = gt.get(name, None)
                if val is None: return -1

            roles = out.values()
            for role in roles:
                if 'knight' not in role.lower() and 'knave' not in role.lower(): return -1

            # formatting of final answer also good
            return 2 if gt == out else -0.5
        
        return [calculate_reward(c, gt) for c, gt in zip(completions, answer)]
    

    SYSTEM_PROMPT = """
    A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
    The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
    The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively.
    i.e., <think> reasoning process here </think> <answer> answer here </answer>.
    User: Please solve this logical reasoning problem. After thinking, when you finally reach a conclusion, clearly state the identity of each character within <answer> </answer> tags i.e., <answer> (1) Zoey is a knight\n(2) ... </answer>. Here is my question,
    
    {prompt}
    Assistant: <think>
    """
    data_path = "/home/VRL/vanilla_trainer/Logic-RL"
    
    ppl_ds = []
    for ppl_file in os.listdir(data_path):
        ds = load_dataset("parquet", data_files=f"{data_path}/{ppl_file}/train.parquet")["train"]
        ds = ds.add_column(name="file", column=[ppl_file]*len(ds))
        ppl_ds.append(ds)
    
    ppl_ds = concatenate_datasets(ppl_ds)
    

    ppl_ds_test = []
    for ppl_file in os.listdir(data_path):
        ds = load_dataset("parquet", data_files=f"{data_path}/{ppl_file}/test.parquet")["train"]
        ds = ds.add_column(name="file", column=[ppl_file]*len(ds))
        ppl_ds_test.append(ds)
    
    ppl_ds_test = concatenate_datasets(ppl_ds_test)


    ppl_ds = ppl_ds.map(lambda x: {
        "prompt": SYSTEM_PROMPT.format(prompt=x["quiz"]),
        "answer": x["solution_text_format"],
        })


    ppl_ds_test = ppl_ds_test.map(lambda x: {
        "prompt": SYSTEM_PROMPT.format(prompt=x["quiz"]),
        "answer": x["solution_text_format"],
        })

    return ppl_ds, ppl_ds_test, reward_kk




def load_humeval(sample=50):
    """Helper function to load HumnEval+ train, test datasets and the reward function.
    Reward function is built upon implementation from OpenAI: https://github.com/openai/human-eval"""
    SYSTEM="""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> final code answer inside </answer>. User: Complete the function based on the given docstring. You must put your final answer (full function with all necessary imports) inside <answer> ```python (code) ```</answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted from within the <answer>```python (code)``` </answer> tags and executed.
    {prompt}
    Assistant: <think>"""
    train = load_dataset("evalplus/humanevalplus", split="test")

    ds = train.train_test_split(train_size=sample, shuffle=True, seed=42)
    train = ds['train']
    test = ds['test']

    def reward_humeval(completions, **kwargs):
        def extract_code(text):
            pattern = r"```python(.+?)```"
            match = re.search(pattern, text, re.DOTALL)
            if match: 
                print("1st extract worked")
                return match.group(1).strip()

            pattern = r"```(.+?)```"
            match = re.search(pattern, text, re.DOTALL)
            if match: 
                print("2nd extract worked")
                return match.group(1).strip()
            
            print("3rd extract")
            return text.strip()

        def calculate_reward(output, problem):
            pattern = r".+</think>\s*<answer>(.+)</answer>\s*$"
            # pattern = r".+</think>\s*<answer>\s*```python\n(.+)```\s*</answer>\s*$"
            s = output 
            if not (s.count("</think>") == 1 and s.count("<answer>") == 1 and s.count("</answer>") == 1):
                # incorrect amount of tokens
                #print("counts no match")
                return -2 
            match = re.search(pattern, s, re.DOTALL)
            # if answer doesn't match provided format
            if not match: return -2

            # answer format is correct now
            #
            result = check_correctness(problem, extract_code(match.group(1).strip()), timeout=30)    
            
            if result['passed']: return 2
            return -1
        
        results = []
        for index in range(len(completions)):
            c = completions[index]
            problem = {
                        'task_id': kwargs['task_id'][index],
                        'prompt': kwargs['prompts'][index],
                        'entry_point': kwargs['entry_point'][index],
                        'test': kwargs['test'][index]
                    }
            results.append(calculate_reward(c, problem))
        
        return results

    train = train.map(lambda x: {
        "prompt": SYSTEM.format(prompt=x["prompt"]),
        })

    test = test.map(lambda x: {
        "prompt": SYSTEM.format(prompt=x["prompt"]),
        })
    
    print(train)
    return train, test, reward_humeval




