import json
from human_eval.evaluation import evaluate_functional_correctness

input_path = "/home/szf/JacobiForcing/eval/CLLM2_eval_generations/baselines/oct_n16w16_distilln32w16_212kstps_greedy_code_only_prompt_humaneval_w_kv_generation_JacobiForcing_Coder_7B_v1.jsonl"
output_path = input_path.replace(".jsonl", "_results.jsonl")

# 转换格式
with open(input_path) as fin, open(output_path, "w") as fout:
    for line in fin:
        item = json.loads(line)
        fout.write(json.dumps({
            "task_id": item["task_id"],
            "completion": item["generation"]
        }) + "\n")

# 评估
results = evaluate_functional_correctness(output_path)
print(results)