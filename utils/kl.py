import torch
import torch.distributed as dist
import torch.nn.functional as F
import pandas as pd
import os
from transformers import AutoModelForCausalLM, AutoTokenizer


PARQUET_PATH = "/mnt/szf_temp/_datasets/openai_humaneval/openai_humaneval/test-00000-of-00001_clean.parquet"
MAX_INPUT_TOKENS = 1024
MAX_NEW_TOKENS = 64
PRINT_EXAMPLES = 5


def setup():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    if world_size != 2:
        raise RuntimeError(f"This script requires exactly 2 processes, got {world_size}.")
    return rank, local_rank


def load_model(name, device):
    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=torch.float16
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(name)
    return model, tokenizer


def forward_logits(model, tokenizer, text, device):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    ).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits  # [1, L, V]
    return logits


def generate_text(model, tokenizer, text, device):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    ).to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = out[0, prompt_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def teacher_decode_sequence(model, tokenizer, prompt_text, device):
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    ).to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = out[0, prompt_len:]
    full_text = tokenizer.decode(out[0], skip_special_tokens=True)
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return full_text, gen_text


def compute_kl(logits_t, logits_s):
    logp_t = F.log_softmax(logits_t, dim=-1)
    logp_s = F.log_softmax(logits_s, dim=-1)
    p_t = logp_t.exp()
    kl = (p_t * (logp_t - logp_s)).sum(dim=-1)  # [1, L]
    return kl.mean()


def main():
    rank, local_rank = setup()
    device = torch.device(f"cuda:{local_rank}")

    df = pd.read_parquet(PARQUET_PATH)
    # first 10 samples for quick test, can use all samples for full evaluation
    df = df.head(10)
    texts = df["prompt"].astype(str).tolist()

    # 👉 可以换成不同模型
    # student_model_name = "/mnt/szf_temp/huggingface/JacobiForcing_Coder_7B_v1"
    student_model_name = "/mnt/szf_temp/huggingface/Qwen2.5-Coder-3B-Instruct"
    teacher_model_name = "/mnt/szf_temp/huggingface/Qwen2.5-Coder-7B-Instruct"

    if rank == 0:
        # ===== GPU0: teacher =====
        model, tokenizer = load_model(teacher_model_name, device)
        for prompt_text in texts:
            # 先用 teacher 解码，再在 teacher 解码出的完整序列上计算 teacher logits
            teacher_seq_text, gen_t = teacher_decode_sequence(model, tokenizer, prompt_text, device)
            logits_t = forward_logits(model, tokenizer, teacher_seq_text, device)

            shape = torch.tensor(logits_t.shape, dtype=torch.long, device=device)
            dist.send(shape, dst=1)
            dist.send(logits_t.contiguous(), dst=1)

            teacher_obj = [teacher_seq_text, gen_t]
            dist.broadcast_object_list(teacher_obj, src=0)

    elif rank == 1:
        # ===== GPU1: student =====
        model, tokenizer = load_model(student_model_name, device)
        kl_sum = 0.0
        kl_count = 0
        exact_match = 0
        examples = []

        for i, prompt_text in enumerate(texts):

            shape = torch.empty(3, dtype=torch.long, device=device)
            dist.recv(shape, src=0)
            shape = tuple(shape.tolist())

            logits_t = torch.empty(shape, dtype=torch.float16, device=device)
            dist.recv(logits_t, src=0)

            teacher_obj = ["", ""]
            dist.broadcast_object_list(teacher_obj, src=0)
            teacher_seq_text, gen_t = teacher_obj

            # 用 teacher 解码出的完整序列做 student prefill，再算 KL
            logits_s = forward_logits(model, tokenizer, teacher_seq_text, device)

            # 保留生成一致性检查：student 仍按原 prompt greedy 生成
            gen_s = generate_text(model, tokenizer, prompt_text, device)

            if logits_t.size(-1) == logits_s.size(-1):
                min_len = min(logits_t.size(1), logits_s.size(1))
                kl = compute_kl(logits_t[:, :min_len, :], logits_s[:, :min_len, :])
                kl_sum += float(kl.item())
                kl_count += 1

            else:
                print(f"[Sample {i}] Skipping KL due to vocab size mismatch: teacher={logits_t.size(-1)}, student={logits_s.size(-1)}")

            same = gen_t.strip() == gen_s.strip()
            if same:
                exact_match += 1

            if len(examples) < PRINT_EXAMPLES:
                examples.append((i, same, gen_t, gen_s))

        avg_kl = kl_sum / kl_count if kl_count > 0 else float("nan")
        ratio = exact_match / len(texts) if len(texts) > 0 else 0.0

        print(f"Total samples: {len(texts)}")
        print(f"KL valid samples: {kl_count}")
        print(f"Average KL(teacher || student): {avg_kl}")
        print(f"Generation exact-match ratio: {ratio:.4f} ({exact_match}/{len(texts)})")
        print("\n=== Output examples ===")
        for i, same, gen_t, gen_s in examples:
            print(f"\n[Sample {i}] same={same}")
            print("Teacher output:")
            print(gen_t)
            print("Student output:")
            print(gen_s)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()