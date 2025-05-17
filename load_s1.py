
import os
import re
import json
import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset, concatenate_datasets

from custom_MATH_reward import compute_score, remove_boxed, last_boxed_only_string

train = load_dataset("simplescaling/s1K_tokenized", split="train")

SYSTEM="""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> final answer inside \\boxed{{}} tag </answer>. 
User: You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted automatically by the \\boxed{{}} tag.
"""

# Assistant: <think>
train = train.to_pandas()

train['text'] = train['text'].str.replace(r"^.*?<\|im_start\|>user", SYSTEM, regex=True, flags=re.DOTALL)
train['text'] = train['text'].str.replace("<|im_start|>assistant", " Assistant: <think>")
train['text'] = train['text'].str.replace("\n<|im_start|>think", "")
train['text'] = train['text'].str.replace("<|im_start|>answer", "")

train['text'] = train['text'].str.replace(
    r'(Final Answer:.*?\$\\boxed\{.*?\}\$)<\|im_end\|>',
    r'</think><answer>\1</answer>',
    regex=True
)

train['text'] = train['text'].str.replace("<|im_end|>", "")
train['text'] = train['text'].str.strip()


def check_pattern(text):
    pattern = r".+</think>\s*<answer>(.+)</answer>\s*$"
    s =text 
    if not (s.count("</think>") == 1 and s.count("<answer>") == 1 and s.count("</answer>") == 1):
        # incorrect amount of tokens
        #print("counts no match")
        return -2 
    match = re.search(pattern, s, re.DOTALL)
    if not match: return -2
    return 0

train['pattern'] = train['text'].apply(check_pattern)

print(train.iloc[0]['text'])

train[['text', 'pattern']].to_csv("s1k_data.csv", index=False)


# print(train['text'][0])
# print("=======")
# print(train['text'][23])
# train = train.map(lambda x: {
#     "prompt": SYSTEM.format(prompt=x["problem"]),
#     "answer": extract_boxed_answer(x["solution"]),
#     "level": x["level"]
#     })

# test = test.map(lambda x: {
#     "prompt": SYSTEM.format(prompt=x["problem"]),
#     "answer": x["answer"],
#     "level": x["level"]
#     })

# if diff_filter:
#     train = train.filter(lambda x: (x['level'] == 'Level 4') or (x['level'] == 'Level 5'))

# train = train.remove_columns(["problem", "solution", "type"])
# test = test.remove_columns(["problem", "solution", "subject", "unique_id"])
