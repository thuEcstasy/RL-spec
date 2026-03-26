from transformers import Qwen3ForCausalLM, AutoTokenizer
from datasets import load_dataset
from einops import rearrange
from torch import nn
import torch.nn.functional as F
import torch
import random
import math
import json
import numpy as np
from tqdm import tqdm
import time
import argparse
import bisect

import os

import pandas as pd

from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from modeling.cllm2_qwen2_modeling_kv_terminate_on_eos_improved import jacobi_forward_greedy
Qwen3ForCausalLM.jacobi_forward_greedy = jacobi_forward_greedy


def parse_args():
    parser = argparse.ArgumentParser(description="HumanEval Jacobi inference with optional oracle window schedule")
    parser.add_argument(
        "--oracle-mode",
        type=str,
        default="off",
        choices=["off", "record", "replay"],
        help="off: normal fixed window; record: save accepted-token counts; replay: derive window from saved counts",
    )
    parser.add_argument(
        "--oracle-trace-path",
        type=str,
        default="oracle_window_humaneval_qwen3.json",
        help="Path to oracle trace json file used by record/replay",
    )
    parser.add_argument(
        "--default-window-size",
        type=int,
        default=64,
        help="Fallback/default window size when oracle is not used or schedule is exhausted",
    )
    parser.add_argument(
        "--max-window-size",
        type=int,
        default=64,
        help="Upper bound for dynamic oracle window size",
    )
    return parser.parse_args()


def clamp_window(window_size, min_window=1, max_window=64):
    return max(min_window, min(int(window_size), max_window))


def get_oracle_schedule(idx, task_id, by_index, by_task_id):
    if str(idx) in by_index:
        return by_index[str(idx)]
    if task_id in by_task_id:
        return by_task_id[task_id]
    return None


def get_window_from_schedule_idx(oracle_schedule, schedule_idx, default_window, max_window, add_next=False):
    if oracle_schedule is None:
        return clamp_window(default_window, max_window=max_window)
    if schedule_idx < 0 or schedule_idx >= len(oracle_schedule):
        return clamp_window(default_window, max_window=max_window)

    target_window = int(oracle_schedule[schedule_idx]) + 12
    # if add_next and (schedule_idx + 1) < len(oracle_schedule):
    #     target_window += int(oracle_schedule[schedule_idx + 1])
    return clamp_window(target_window, max_window=max_window)


def build_flat_oracle_trace(oracle_token_id_schedule):
    flat_tokens = []
    token_pos_to_call_idx = []
    if oracle_token_id_schedule is None:
        return flat_tokens, token_pos_to_call_idx

    for call_idx, token_ids in enumerate(oracle_token_id_schedule):
        for token_id in token_ids:
            flat_tokens.append(int(token_id))
            token_pos_to_call_idx.append(call_idx)
    return flat_tokens, token_pos_to_call_idx


def _bigram_key(prev_token, curr_token):
    return f"{int(prev_token)}|{int(curr_token)}"


def build_bigram_token_pos_index(flat_tokens):
    index = {}
    for pos in range(1, len(flat_tokens)):
        key = _bigram_key(flat_tokens[pos - 1], flat_tokens[pos])
        if key not in index:
            index[key] = []
        index[key].append(pos)
    return index


def find_schedule_idx_by_bigram(bigram_token_pos_index, token_pos_to_call_idx, prev_token, curr_token, start_pos):
    if len(token_pos_to_call_idx) < 2:
        return None, start_pos

    positions = bigram_token_pos_index.get(_bigram_key(prev_token, curr_token))
    if not positions:
        return None, start_pos

    scan_start = max(1, int(start_pos))
    pos_idx = bisect.bisect_left(positions, scan_start)
    if pos_idx >= len(positions):
        return None, start_pos

    matched_pos = positions[pos_idx]
    if matched_pos >= len(token_pos_to_call_idx):
        return None, start_pos
    return token_pos_to_call_idx[matched_pos], matched_pos + 1


def resize_draft_to_window(draft_ids, target_window, token_pool):
    """Pad/truncate draft tokens so jacobi_forward_greedy sees the target draft size."""
    if draft_ids.shape[1] == target_window:
        return draft_ids
    if draft_ids.shape[1] > target_window:
        return draft_ids[:, :target_window]

    pad_len = target_window - draft_ids.shape[1]
    if (token_pool is not None) and (token_pool.numel() > 0):
        rand_idx = torch.randint(0, token_pool.shape[1], (pad_len,), device=draft_ids.device)
        pad_tokens = token_pool[:, rand_idx]
    else:
        pad_tokens = draft_ids[:, -1:].repeat(1, pad_len)
    return torch.cat((draft_ids, pad_tokens), dim=-1)


args = parse_args()


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# Load dataset
df = pd.read_parquet("/home/szf/datasets/openai_humaneval/openai_humaneval/test-00000-of-00001_clean.parquet")
df_size = len(df)
print(f"Loaded HumanEval dataset with {df_size} samples")
records = df.to_dict(orient="records")

# ---------------------------
# Load model/tokenizer once
# ---------------------------
# model_name = "/home/szf/huggingface/JacobiForcing_Coder_7B_v1"
model_name = "/home/szf/huggingface/Qwen3-1.7B"

model = Qwen3ForCausalLM.from_pretrained(
    model_name,
    device_map="cuda",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)
tokenizer = AutoTokenizer.from_pretrained("/home/szf/huggingface/Qwen3-1.7B")
model.eval()


eos_id = tokenizer.eos_token_id
alt_eos_id = 151645  # keep your special EOS as a fallback

# ---------------------------
# Generation/profiling config
# ---------------------------
seed = 42
set_seed(seed)
print(f"Using seed={seed}")

default_n_token_seq_len = clamp_window(args.default_window_size, max_window=args.max_window_size)
print(
    f"Oracle mode={args.oracle_mode}, default_window={default_n_token_seq_len}, "
    f"max_window={args.max_window_size}, trace_path={args.oracle_trace_path}"
)

oracle_by_index = {}
oracle_by_task_id = {}
oracle_token_ids_by_index = {}
oracle_token_ids_by_task_id = {}
oracle_bigram_index_by_index = {}
oracle_bigram_index_by_task_id = {}
if args.oracle_mode == "replay":
    with open(args.oracle_trace_path, "r") as f:
        oracle_payload = json.load(f)
    oracle_by_index = oracle_payload.get("by_index", {})
    oracle_by_task_id = oracle_payload.get("by_task_id", {})
    oracle_token_ids_by_index = oracle_payload.get("by_index_token_ids", {})
    oracle_token_ids_by_task_id = oracle_payload.get("by_task_id_token_ids", {})
    oracle_bigram_index_by_index = oracle_payload.get("by_index_bigram_token_pos_index", {})
    oracle_bigram_index_by_task_id = oracle_payload.get("by_task_id_bigram_token_pos_index", {})
    print(
        f"Loaded oracle schedule: {len(oracle_by_index)} index entries, "
        f"{len(oracle_by_task_id)} task_id entries"
    )
    print(
        f"Loaded oracle token-id trace: {len(oracle_token_ids_by_index)} index entries, "
        f"{len(oracle_token_ids_by_task_id)} task_id entries"
    )
    print(
        f"Loaded oracle bigram index: {len(oracle_bigram_index_by_index)} index entries, "
        f"{len(oracle_bigram_index_by_task_id)} task_id entries"
    )
    print("Replay window strategy: match (prev_token, curr_token) in trace -> window_i = accept_i + accept_{i+1}; fallback=default")

oracle_records = []

# Safety caps so a sample can't run forever.
max_new_tokens = 32768     # hard cap on total new tokens per prompt
max_calls = 1024          # hard cap on number of diffusion_decoding calls per prompt

# ---------------------------
# Iterate the dataset
# ---------------------------
all_rows = []
t0_overall = time.perf_counter()
all_generations = []

total_gen_only_time = 0
global_profile_call_index = 0

for idx, row in tqdm(enumerate(records)):
    task_id = row.get("task_id", f"idx_{idx}")
    #prompt = "You are given a partially completed Python function with the header and the doc string. Complete the following function according to given information:\n\n" + row["prompt"]
    prompt = """
Please continue to complete the function. You are not allowed to modify the given code and do the completion only. Please return all completed function in a codeblock. Here is the given code to do completion:
```python
{}
```
""".strip().format(
            row["prompt"].strip()
        )

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    input_ids = model_inputs["input_ids"]
    attention_mask = torch.full_like(input_ids, 1, device=model.device)

    # per-example stats
    iters = []
    total_new_tokens = 0
    calls = 0
    model._jacobi_draft = None  # reset warm draft at start of each example
    prev_len = input_ids.shape[1]
    prompt_len = prev_len
    stop_reason = None
    prefill_phase = True
    generated_ids = input_ids
    
    prefill_drafted_n_gram = None
    accepted_tokens_per_call = []
    accepted_token_ids_per_call = []
    oracle_schedule = get_oracle_schedule(idx, task_id, oracle_by_index, oracle_by_task_id)
    oracle_token_id_schedule = get_oracle_schedule(idx, task_id, oracle_token_ids_by_index, oracle_token_ids_by_task_id)
    oracle_bigram_token_pos_index = get_oracle_schedule(
        idx,
        task_id,
        oracle_bigram_index_by_index,
        oracle_bigram_index_by_task_id,
    )
    oracle_flat_tokens, oracle_token_pos_to_call_idx = build_flat_oracle_trace(oracle_token_id_schedule)
    if oracle_bigram_token_pos_index is None:
        oracle_bigram_token_pos_index = build_bigram_token_pos_index(oracle_flat_tokens)
    else:
        oracle_bigram_token_pos_index = {
            str(key): [int(pos) for pos in positions]
            for key, positions in oracle_bigram_token_pos_index.items()
        }
    oracle_search_pos = 1

    replay_window_bigram_hit_calls = 0
    replay_window_bigram_miss_calls = 0
    
    gen_only_time = 0

    t_start = time.time()
    # run until EOS or caps
    while True:
        # Check EOS
        generated_part = generated_ids[0, prompt_len:]
        hit_eos = False
        if eos_id is not None:
            hit_eos = (generated_part == eos_id).any().item()
        if not hit_eos:
            # allow alternate special EOS id
            hit_eos = (generated_part == alt_eos_id).any().item()

        if hit_eos:
            stop_reason = "eos"
            break
        if total_new_tokens >= max_new_tokens:
            stop_reason = "max_new_tokens"
            break
        if calls >= max_calls:
            stop_reason = "max_calls"
            break
        
        #print(f"\nInit new subsequence {calls}...\n")

        ### One diffusion decoding call
        sample_call_index = calls + 1
        if prefill_phase:
            # Keep prefill deterministic and independent from oracle replay.
            # Oracle schedule is recorded from generation calls only.
            prefill_window = default_n_token_seq_len
            # pass in random-init draft
            q_sampled = []
            for _ in range(prefill_window):
                q_sample = torch.tensor([random.choice(generated_ids[0].tolist())], dtype=torch.long, device=model.device).unsqueeze(0)
                q_sampled.append(q_sample)
            prefill_draft_token_ids = torch.cat(q_sampled, dim=1)  # shape [1, prefill_window]
            
            prefill_input_ids = torch.cat((input_ids, prefill_draft_token_ids),dim=-1)

            global_profile_call_index += 1
            
            # `jacobi_forward_greedy` will return iteration result from first iteration
            past_key_values, first_correct_token, prefill_drafted_n_gram, iter_count = model.jacobi_forward_greedy(
                input_ids=prefill_input_ids,
                attention_mask=attention_mask,
                past_key_values=None,
                use_cache=True,
                prefill_phase=prefill_phase,
                n_token_seq_len=prefill_window,
                tokenizer=tokenizer,
                fixed_window=True,
                eos_token_id=eos_id,
                profile_sample_index=idx,
                profile_call_index=sample_call_index,
                profile_global_call_index=global_profile_call_index,
                )
            prefill_phase = False
            generated_ids = input_ids
            itr_count = 0
        else:
            target_window = default_n_token_seq_len
            if args.oracle_mode == "replay":
                generated_part = generated_ids[0, prompt_len:]
                if generated_part.shape[0] >= 2:
                    prev_token = int(generated_part[-2].item())
                    curr_token = int(generated_part[-1].item())
                    matched_schedule_idx, oracle_search_pos = find_schedule_idx_by_bigram(
                        bigram_token_pos_index=oracle_bigram_token_pos_index,
                        token_pos_to_call_idx=oracle_token_pos_to_call_idx,
                        prev_token=prev_token,
                        curr_token=curr_token,
                        start_pos=oracle_search_pos,
                    )
                    if matched_schedule_idx is not None:
                        target_window = get_window_from_schedule_idx(
                            oracle_schedule=oracle_schedule,
                            schedule_idx=matched_schedule_idx,
                            default_window=default_n_token_seq_len,
                            max_window=args.max_window_size,
                            add_next=True,
                        )
                        replay_window_bigram_hit_calls += 1
                    else:
                        replay_window_bigram_miss_calls += 1
                else:
                    replay_window_bigram_miss_calls += 1
            # generation phase
            # ---- Initialize a draft tail (any tokens work; we'll fix on the first pass).
            # We keep your "random from prompt" init to avoid extra forward passes.
            if calls == 1:
                # First non-prefill call: reuse draft_tokens produced by prefill
                input_ids = prefill_drafted_n_gram
            else:
                if model._jacobi_draft is not None:
                    # Reuse the model's own warm predictions from the last call.
                    # This lets Jacobi iteration converge across calls.
                    input_ids = model._jacobi_draft
                else:
                    # Fallback (e.g. all-accepted case): GPU-native random init
                    if target_window <= 1:
                        input_ids = first_correct_token.view(1, 1)
                    else:
                        rand_idx = torch.randint(0, generated_ids.shape[1], (target_window - 1,), device=model.device)
                        q_sampled = generated_ids[:, rand_idx]
                        input_ids = torch.cat((first_correct_token.view(1, 1), q_sampled), dim=-1)

            input_ids = resize_draft_to_window(
                draft_ids=input_ids,
                target_window=target_window,
                token_pool=generated_ids,
            )

            t_gen_start = time.perf_counter()
            accepted_history_ids = generated_ids[:, prompt_len:]
            global_profile_call_index += 1
            past_key_values, first_correct_token, accepted_n_gram, itr_count = model.jacobi_forward_greedy(
                input_ids=input_ids,
                attention_mask=None,
                past_key_values=past_key_values,
                use_cache=True,
                prefill_phase=prefill_phase,
                n_token_seq_len=target_window,
                tokenizer=tokenizer,
                accepted_history_ids=accepted_history_ids,
                fixed_window=True,
                eos_token_id=eos_id,
                profile_sample_index=idx,
                profile_call_index=sample_call_index,
                profile_global_call_index=global_profile_call_index,
            )
            t_gen_time = time.perf_counter() - t_gen_start
            gen_only_time += t_gen_time
            
            generated_ids = torch.cat((generated_ids, accepted_n_gram), dim=-1)
            accepted_token_ids = accepted_n_gram[0].tolist()
            accepted_tokens_per_call.append(len(accepted_token_ids))
            accepted_token_ids_per_call.append(accepted_token_ids)

        calls += 1
        iters.append(itr_count)

        added = generated_ids.shape[1] - prev_len
        if added > 0:
            total_new_tokens += added
        prev_len = generated_ids.shape[1]
    
    # subtract prefill
    total_new_tokens -= 1
    # per-example finalize
    dt = time.time() - t_start
    total_iterations = sum(iters)
    avg_iter_per_call = (total_iterations / calls)
    avg_iter_per_token = (total_iterations / total_new_tokens)
    
    toks_per_sec = (total_new_tokens / gen_only_time)
    
    total_gen_only_time += gen_only_time
    
    prompt_len = model_inputs["input_ids"].shape[1]
    generated_str = ''.join(tokenizer.decode(generated_ids[0, prompt_len:], skip_special_tokens=False))
    print(f'Generated answers: {generated_str}')
    all_generations.append(generated_str)

    all_rows.append(
        {
            "index": idx,
            "task_id": task_id,
            "prompt_tokens": prompt_len,
            "new_tokens": total_new_tokens,
            "calls": calls,
            "total_iterations": total_iterations,
            "avg_iter_per_call": avg_iter_per_call,
            "avg_iter_per_token": avg_iter_per_token,
            "time_sec": dt,
            "toks_per_sec": toks_per_sec,
            "stop_reason": stop_reason,
            "accepted_tokens_per_call": accepted_tokens_per_call,
            "replay_window_bigram_hit_calls": replay_window_bigram_hit_calls if args.oracle_mode == "replay" else None,
            "replay_window_bigram_miss_calls": replay_window_bigram_miss_calls if args.oracle_mode == "replay" else None,
        }
    )

    if args.oracle_mode == "record":
        record_flat_tokens, _ = build_flat_oracle_trace(accepted_token_ids_per_call)
        record_bigram_index = build_bigram_token_pos_index(record_flat_tokens)
        oracle_records.append(
            {
                "index": idx,
                "task_id": task_id,
                "accepted_tokens_per_call": accepted_tokens_per_call,
                "accepted_token_ids_per_call": accepted_token_ids_per_call,
                "bigram_token_pos_index": record_bigram_index,
            }
        )

    # light progress
    if (idx + 1) % 5 == 0 or (idx + 1) == len(records):
        print(f"====[{idx+1}/{len(records)}] task_id={task_id} new_toks={total_new_tokens} "
              f"calls={calls} avg_iter/call={avg_iter_per_call:.2f} reason={stop_reason}====")
        
#### ADDED Lines ####
import re

# Function to load the data from JSONL
def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line.strip()) for line in f]

# Function to save the data to JSONL
def save_jsonl(data, save_path):
    with open(save_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

# Function to extract Python code block from a string
def extract_python_code(text):
    match = re.search(r'```python([\s\S]*?)```', text)  # Regex to match the block
    if match:
        return match.group(1).strip()  # Return the code inside the block
    else:
        return text  # Return orginal one if no match is found

eval_dir = "/home/szf/JacobiForcing/eval/CLLM2_eval_generations/baselines_qwen3"
os.makedirs(eval_dir, exist_ok=True)

original_path = os.path.join(eval_dir, 'humaneval_python_example_clean.jsonl')
original_generations = load_jsonl(original_path)

# Process each generation and update with processed generation
for i, original_generation in enumerate(original_generations):
    # Assuming `all_generations[i]` exists and has an 'extracted' key or method
    original_generation['output'] = all_generations[i]
    processed_generation = extract_python_code(all_generations[i])  # Apply the extract method
    print(f'Task id: {i}, Extracted answer: {processed_generation}')
    original_generation['generation'] = processed_generation

# Save processed generations
save_path = os.path.join(
    eval_dir,
    f'slidingw_{args.oracle_mode}_defaultw{default_n_token_seq_len}_'
    f'code_only_prompt_humaneval_w_kv_generation_{model_name.split("/")[-1]}.jsonl'
)
save_jsonl(original_generations, save_path)

print(f"\n=== All generation done (HumanEval). Results are saved to {save_path} ===")

#### ADDED Lines ####

# ---------------------------
# Aggregate + save
# ---------------------------
t_overall = time.perf_counter() - t0_overall
df_profile = pd.DataFrame(all_rows)
csv_path = "diffusion_profile_humaneval.csv"
df_profile.to_csv(csv_path, index=False)

# Print quick summary (EOS-only)
def _safe_mean(series):
    s = pd.to_numeric(series, errors="coerce")
    return float(s.mean()) if s.size and not pd.isna(s).all() else float("nan")

df_eos = df_profile[df_profile["stop_reason"] == "eos"].copy()
n_eos = len(df_eos)
n_total = len(df_profile)

print("\n=== Diffusion Decoding Profiling — EOS-only ===")
print(f"Examples (eos): {n_eos} / {n_total}   Total wall time: {t_overall:.4f}s")
print(f"Avg new tokens / prompt: {_safe_mean(df_eos['new_tokens']):.4f}")
print(f"Avg calls / prompt: {_safe_mean(df_eos['calls']):.4f}")
print(f"Avg iterations / call: {_safe_mean(df_eos['avg_iter_per_call']):.4f}")
avg_iter_per_token = _safe_mean(df_eos['avg_iter_per_token'])
avg_token_per_iter = (1.0 / avg_iter_per_token) if (math.isfinite(avg_iter_per_token) and avg_iter_per_token != 0.0) else float("nan")
print(f"Avg iterations / token: {avg_iter_per_token:.4f}")
print(f"Avg tokens / iteration: {avg_token_per_iter:.4f}")
print(f"Avg toks/sec: {_safe_mean(df_eos['toks_per_sec']):.4f}")

# Optional: also show overall stop-reason distribution for context
print("\nStop reasons (all examples):")
print(df_profile['stop_reason'].value_counts())

# Optional: save EOS-only rows too
df_eos.to_csv("diffusion_profile_greedy_humaneval_eos_qwen3.csv", index=False)

if args.oracle_mode == "record":
    oracle_payload = {
        "version": 1,
        "dataset_path": "/home/szf/datasets/openai_humaneval/openai_humaneval/test-00000-of-00001_clean.parquet",
        "default_window_size": default_n_token_seq_len,
        "max_window_size": args.max_window_size,
        "num_examples": len(oracle_records),
        "by_index": {str(item["index"]): item["accepted_tokens_per_call"] for item in oracle_records},
        "by_task_id": {item["task_id"]: item["accepted_tokens_per_call"] for item in oracle_records},
        "by_index_token_ids": {str(item["index"]): item["accepted_token_ids_per_call"] for item in oracle_records},
        "by_task_id_token_ids": {item["task_id"]: item["accepted_token_ids_per_call"] for item in oracle_records},
        "by_index_bigram_token_pos_index": {str(item["index"]): item["bigram_token_pos_index"] for item in oracle_records},
        "by_task_id_bigram_token_pos_index": {item["task_id"]: item["bigram_token_pos_index"] for item in oracle_records},
        "records": oracle_records,
    }
    with open(args.oracle_trace_path, "w") as f:
        json.dump(oracle_payload, f, indent=2)
    print(f"Saved oracle trace to {args.oracle_trace_path}")

if args.oracle_mode == "replay":
    total_bigram_hit_calls = int(pd.to_numeric(df_profile["replay_window_bigram_hit_calls"], errors="coerce").fillna(0).sum())
    total_bigram_miss_calls = int(pd.to_numeric(df_profile["replay_window_bigram_miss_calls"], errors="coerce").fillna(0).sum())
    print("\n=== Oracle Token-ID Replay Validation ===")
    print(f"Window bigram-hit calls: {total_bigram_hit_calls}")
    print(f"Window bigram-miss calls: {total_bigram_miss_calls}")
