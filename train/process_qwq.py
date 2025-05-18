import pandas as pd
import re
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B")

from dataset_loader import load_kk


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

#df = pd.read_csv("qwq_samples_more.csv")
# df = pd.read_csv("graph_qwq_samples.csv")
# df = pd.read_csv("s1k_data.csv")
df = pd.read_csv("kk_qwen32_samples.csv")
print(df.columns)

kk_train, kk_test, reward_kk = load_kk()

kk_answers = list(kk_train.to_pandas()['answer'].repeat(4))

df['llm_answer'] = df['llm_answer'] + "</answer>"
# df['pattern'] = [check_pattern(text) for text in df['llm_answer']]
df['pattern'] = [reward_kk([text], [kk_answers[index]])[0] for index, text in enumerate(df['llm_answer'])]
df['tokens'] = [len(tokenizer(text)['input_ids']) for text in df['llm_answer']]



#df.to_csv("processed_qwq_samples.csv", index=False)
#df.to_csv("processed_graph_qwq_samples.csv", index=False)
df.to_csv("rs_processed_qwen32_samples.csv", index=False)
