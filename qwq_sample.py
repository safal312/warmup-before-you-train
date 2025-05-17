import os
from vllm import LLM, SamplingParams
import pandas as pd
from datasets import load_dataset


from dataset_loader import load_math, load_countdown, load_kk, load_graph_data


SYSTEM="""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> final answer inside \\boxed{{}} tag </answer>. User: You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted automatically by the \\boxed{{}} tag.
{prompt}
Assistant: <think>"""

train, test, _ = load_kk()
#train, test, _ = load_graph_data()

gpus=4
sampling_params = SamplingParams(n=4, temperature=0.6, top_p=0.95, top_k=40, max_tokens=8192, stop=["</answer>", "User:", "Assistant:"])
llm = LLM(model="Qwen/Qwen2.5-32B", max_model_len=9000, gpu_memory_utilization=0.9, tensor_parallel_size=gpus)

process_in = train['prompt']

df = pd.DataFrame()
df['question'] =process_in
df = df.loc[df.index.repeat(gpus)].reset_index(drop=True)

outputs = llm.generate(process_in, sampling_params)

process_out = []

for output in outputs:
    for out in output.outputs:
        process_out.append(out.text)
#process_out = [output.outputs[0].text for output in outputs]


df['llm_answer'] = process_out

#df.to_csv("qwq_samples_more.csv", index=False)
# df.to_csv("graph_qwq_samples.csv", index=False)
df.to_csv("kk_qwen32_samples.csv", index=False)



