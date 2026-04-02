import os
import sys
import random
import argparse
from pathlib import Path

import torch
from transformers import Qwen2ForCausalLM, AutoTokenizer

import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from modeling.cllm2_qwen2_modeling_kv_terminate_on_eos_improved import jacobi_forward_greedy

Qwen2ForCausalLM.jacobi_forward_greedy = jacobi_forward_greedy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="/mnt/szf_temp/huggingface/JacobiForcing_Coder_7B_v1",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="/mnt/szf_temp/huggingface/JacobiForcing_Coder_7B_v1",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
    )
    parser.add_argument(
        "--attn-impl",
        type=str,
        default="flash_attention_2",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Implement quicksort algorithm in Python",
        help="直接传入 prompt 文本",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="从文件读取 prompt",
    )
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def clamp_window(window_size: int, min_window: int = 1, max_window: int = 128):
    return max(min_window, min(int(window_size), max_window))


def resize_draft_to_window(draft_ids: torch.Tensor, target_window: int, token_pool: torch.Tensor):
    if draft_ids.shape[1] == target_window:
        return draft_ids
    if draft_ids.shape[1] > target_window:
        return draft_ids[:, :target_window]

    pad_len = target_window - draft_ids.shape[1]
    if token_pool is not None and token_pool.numel() > 0:
        rand_idx = torch.randint(0, token_pool.shape[1], (pad_len,), device=draft_ids.device)
        pad_tokens = token_pool[:, rand_idx]
    else:
        pad_tokens = draft_ids[:, -1:].repeat(1, pad_len)
    return torch.cat((draft_ids, pad_tokens), dim=-1)


def build_prompt(user_input: str):
    return f"""Please continue to complete the function. You are not allowed to modify the given code and do the completion only. Please return all completed function in a codeblock. Here is the given code to do completion:
```python
{user_input}
```"""


@torch.inference_mode()
def stream_generate_terminal(
    model,
    tokenizer,
    prompt: str,
    window_size: int = 32,
    max_new_tokens: int = 1024,
):
    window_size = clamp_window(window_size, 1, 128)

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    input_ids = model_inputs["input_ids"]
    attention_mask = torch.ones_like(input_ids, device=model.device)

    eos_id = tokenizer.eos_token_id
    alt_eos_id = 151645  # 你原脚本里的 fallback eos id

    generated_ids = input_ids
    prompt_len = input_ids.shape[1]

    total_new_tokens = 0
    calls = 0
    prefill_phase = True

    past_key_values = None
    prefill_drafted_n_gram = None
    first_correct_token = None

    model._jacobi_draft = None

    while True:
        generated_part = generated_ids[0, prompt_len:]

        hit_eos = False
        if eos_id is not None:
            hit_eos = (generated_part == eos_id).any().item()
        if not hit_eos:
            hit_eos = (generated_part == alt_eos_id).any().item()

        if hit_eos:
            break
        if total_new_tokens >= max_new_tokens:
            break

        if prefill_phase:
            # 和你的原逻辑一致：prefill 先随机初始化 draft
            q_sampled = []
            for _ in range(window_size):
                q_sample = torch.tensor(
                    [random.choice(generated_ids[0].tolist())],
                    dtype=torch.long,
                    device=model.device,
                ).unsqueeze(0)
                q_sampled.append(q_sample)

            prefill_draft_token_ids = torch.cat(q_sampled, dim=1)
            prefill_input_ids = torch.cat((input_ids, prefill_draft_token_ids), dim=-1)

            past_key_values, first_correct_token, prefill_drafted_n_gram, iter_count, _ = model.jacobi_forward_greedy(
                input_ids=prefill_input_ids,
                attention_mask=attention_mask,
                past_key_values=None,
                use_cache=True,
                prefill_phase=True,
                n_token_seq_len=window_size,
                tokenizer=tokenizer,
                fixed_window=True,
                eos_token_id=eos_id,
            )

            prefill_phase = False
            generated_ids = input_ids
        else:
            if calls == 1:
                draft_ids = prefill_drafted_n_gram
            else:
                if getattr(model, "_jacobi_draft", None) is not None:
                    draft_ids = model._jacobi_draft
                else:
                    if window_size <= 1:
                        draft_ids = first_correct_token.view(1, 1)
                    else:
                        rand_idx = torch.randint(
                            0,
                            generated_ids.shape[1],
                            (window_size - 1,),
                            device=model.device,
                        )
                        q_sampled = generated_ids[:, rand_idx]
                        draft_ids = torch.cat((first_correct_token.view(1, 1), q_sampled), dim=-1)

            draft_ids = resize_draft_to_window(
                draft_ids=draft_ids,
                target_window=window_size,
                token_pool=generated_ids,
            )

            accepted_history_ids = generated_ids[:, prompt_len:]

            past_key_values, first_correct_token, accepted_n_gram, itr_count, _ = model.jacobi_forward_greedy(
                input_ids=draft_ids,
                attention_mask=None,
                past_key_values=past_key_values,
                use_cache=True,
                prefill_phase=False,
                n_token_seq_len=window_size,
                tokenizer=tokenizer,
                accepted_history_ids=accepted_history_ids,
                fixed_window=True,
                eos_token_id=eos_id,
            )

            generated_ids = torch.cat((generated_ids, accepted_n_gram), dim=-1)
            total_new_tokens = generated_ids.shape[1] - prompt_len

            delta_text = tokenizer.decode(
                accepted_n_gram[0],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )

            print(delta_text, end="", flush=True)

        calls += 1
        import time
        # sleep for 0.5s
        time.sleep(0.5)
    print("\n\n===== Done =====")
    print(f"Total accepted new tokens: {total_new_tokens}")
    print(f"Total calls: {calls}")
    print()  # 最后补一个换行


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    if args.prompt is not None:
        raw_input_prompt = args.prompt
    elif args.prompt_file is not None:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            raw_input_prompt = f.read()
    else:
        print("请输入要补全的代码，按 Ctrl-D 结束：")
        raw_input_prompt = sys.stdin.read()

    final_prompt = build_prompt(raw_input_prompt)

    print(f"Loading model from: {args.model_path}", file=sys.stderr)
    print(f"Loading tokenizer from: {args.tokenizer_path}", file=sys.stderr)

    model = Qwen2ForCausalLM.from_pretrained(
        args.model_path,
        device_map="cuda",
        torch_dtype=torch_dtype,
        attn_implementation=args.attn_impl,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model.eval()

    print("\n===== Streaming output =====\n")
    stream_generate_terminal(
        model=model,
        tokenizer=tokenizer,
        prompt=final_prompt,
        window_size=args.window_size,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()