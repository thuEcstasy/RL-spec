from pathlib import Path
import sys
from datasets import load_dataset
import transformers
import torch

from torch.cuda.amp import autocast
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother

import torch.nn.functional as F

from torch.nn.attention.flex_attention import create_block_mask

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from train.soft_flexattn_train_rl_spec import make_online_jacobi_data_module 
from train.soft_flexattn_train_rl_spec import ModelArguments, DataArguments, TrainingArguments
data_path = "/mnt/szf_temp/datasets/OpenCodeInstruct/data/first_10000.jsonl"
rollout_model_path = "/mnt/szf_temp/huggingface/JacobiForcing_Coder_7B_v1"
model_path = "/mnt/szf_temp/huggingface/JacobiForcing_Coder_7B_v1"
# rollout_model_path = "/mnt/szf_temp/huggingface/Qwen2.5-Coder-7B-Instruct"
# model_path = "/mnt/szf_temp/huggingface/Qwen2.5-Coder-7B-Instruct"
ref_model_path = "/mnt/szf_temp/huggingface/Qwen2.5-Coder-7B-Instruct"
input_path = "/mnt/szf_temp/datasets/OpenCodeInstruct/data/first_10000.jsonl"

parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()

raw_dataset = load_dataset(
    "json",
    data_files={"train": data_path},
    split="train",
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,  # important for debugging / avoid weird empty shards
    attn_implementation="flex_attention",
)
model.to("cuda")


rollout_model = transformers.AutoModelForCausalLM.from_pretrained(
    rollout_model_path,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,  # important for debugging / avoid weird empty shards
)
rollout_model.to("cuda:1")
rollout_model.eval()

ref_model = transformers.AutoModelForCausalLM.from_pretrained(
    ref_model_path,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
)
ref_model.to("cuda:1")
ref_model.eval()

tokenizer = transformers.AutoTokenizer.from_pretrained(
    rollout_model_path,
    padding_side="right",
    use_fast=False,
)

data_module = make_online_jacobi_data_module(
    tokenizer=tokenizer,
    prompt_data=raw_dataset,
    training_args=training_args,
)
dataset = data_module["train_dataset"]
dataset.set_rollout_model(rollout_model)


# load the first line as the input
for i, sample in enumerate(dataset):
    dataset.cfg.n_token_seq_len = 32
    output = dataset._build_training_sample(sample["prompt_ids"])
    
    @staticmethod
    def _to_int(x):
        return x.item() if isinstance(x, torch.Tensor) else int(x)
    def _unpack_sample(inputs):
        """
        Extract a single sample. (Assumes per_device_train_batch_size == 1.)
        Required keys:
          - input_ids: [1, L]
          - prompt_ids_len: scalar or [1]
          - T: length of traj_position_indices (last uncorrupted token positions) in [1, T]
        """
        # TODO: support bsz > 1 uppacking
        input_ids = inputs["input_ids"]
        prompt_len = inputs["prompt_ids_len"]
        if isinstance(prompt_len, torch.Tensor):
            if prompt_len.dim() > 0:
                prompt_len = prompt_len[0]
        prompt_len = _to_int(prompt_len)

        traj_position_indices = inputs["traj_position_indices"][0]
        traj_position_indices = [int(u) for u in traj_position_indices]
        T = len(traj_position_indices)

        return (
            input_ids.to("cuda"),
            prompt_len,
            T,
        )
    @staticmethod
    def _index_layout(prompt_len: int, T: int, N):
        """Return lists of start indices for all k_j and last_j blocks in flattened sequence."""
        k_starts = [prompt_len + 2 * j * N for j in range(T)]
        l_starts = [prompt_len + (2 * j + 1) * N for j in range(T)]
        return k_starts, l_starts
    
    def _flip_block_after_eos_to_pad(input_ids: torch.Tensor, start: int, N: int, eos_id: int | None, pad_id: int | None) -> int:
        """Mutate input_ids[start:start+N] so that tokens AFTER first EOS become PAD.
        Returns number of tokens flipped."""
        if eos_id is None or pad_id is None:
            return 0
        block = input_ids[start:start+N]
        pos = (block == eos_id).nonzero(as_tuple=False)
        if pos.numel() == 0:
            return 0
        k = int(pos[0])                  # eos offset inside block
        flip_start = start + k + 1
        flip_end   = start + N
        if flip_start < flip_end:
            input_ids[flip_start:flip_end] = pad_id
            return flip_end - flip_start
        return 0
    
    def _build_shared_position_ids(L: int, prompt_len: int, T: int):
        """
        Build [L] position_ids so each (k_j, last_j) share the same positions.
        Prompt: 0...prompt_len-1
        For each j: both blocks use prompt_len + j*N .. prompt_len + j*N + (N-1)
        """
        device = "cuda"
        N = 32
        pos = torch.empty(L, dtype=torch.long, device=device)

        # Prompt positions
        pos[:prompt_len] = torch.arange(prompt_len, device=device)

        # Pair-shared positions
        k_starts, l_starts = _index_layout(prompt_len, T, N)
        rel = torch.arange(N, device=device)
        for j in range(T):
            base = prompt_len + j * N
            ks = k_starts[j]
            ls = l_starts[j]
            pos[ks:ks + N] = base + rel
            pos[ls:ls + N] = base + rel

        return pos
    
    def soft_cross_entropy(predicts, targets, padding_mask):
        if (~padding_mask).sum() == 0:
            return 0 * predicts[0][0]
        predict_log_prob = torch.nn.functional.log_softmax(predicts, dim=-1)
        targets_prob = torch.nn.functional.softmax(targets, dim=-1)
        entropy = -targets_prob * predict_log_prob
        expand_mask = padding_mask.unsqueeze(-1).expand_as(entropy)
        entropy = entropy.masked_fill(expand_mask, 0)
        mean_entropy = entropy.sum() / (~padding_mask).sum()
        return mean_entropy
    
    def _build_block_mask(L: int, prompt_len: int, T: int, heads: int,
                          mode: str = "same"):
        """
        Build a BlockMask for the interleaved [k_0, last_0, k_1, last_1, ...] layout.

        mode:
          "same"  – (variant A) k_j attends to prev k_*, last_j attends to prev last_*
          "cross" – (variant B) k_j attends to prev last_* (noisy conditioned on prev clean),
                                last_j still attends to prev last_*
        """
        N = 32
        k_starts, l_starts = _index_layout(prompt_len, T, N)

        ks = torch.tensor(k_starts, device="cuda")
        ls = torch.tensor(l_starts, device="cuda")

        _mode_is_cross = (mode == "cross")

        def mask_mod(b, h, q, k):
            rel_q = q - prompt_len
            rel_k = k - prompt_len
            block_idx_q = torch.div(rel_q, N, rounding_mode="floor")
            block_idx_k = torch.div(rel_k, N, rounding_mode="floor")

            is_prompt_q = q < prompt_len
            is_prompt_k = k < prompt_len

            is_kj_q    = (q >= prompt_len) & (block_idx_q % 2 == 0)
            is_lastj_q = (q >= prompt_len) & (block_idx_q % 2 == 1)
            is_kj_k    = (k >= prompt_len) & (block_idx_k % 2 == 0)
            is_lastj_k = (k >= prompt_len) & (block_idx_k % 2 == 1)

            # j index for q, clamped to [0, T-1]
            j_q = torch.clamp(block_idx_q // 2, min=0, max=T - 1)

            ks_per_q = ks[j_q]
            ls_per_q = ls[j_q]

            # ---- what previous blocks does k_j attend to? ----
            if _mode_is_cross:
                # variant B: k_j attends to all previous last_* blocks
                # (last blocks have odd block indices: 1, 3, 5, ... 2*(j_q-1)+1)
                kj_prev = is_lastj_k & (block_idx_k < 2 * j_q)
            else:
                # variant A: k_j attends to all previous k_* blocks
                kj_prev = is_kj_k & (block_idx_k < 2 * j_q)

            # last_j always attends to previous last_* (unchanged in both modes)
            last_in_prev_last = is_lastj_k & (block_idx_k < 2 * j_q)

            # prompt is always causal
            mask_prompt = is_prompt_q & (k <= q)

            # k_j queries:
            same_kj_block = is_kj_q & is_kj_k & (block_idx_q == block_idx_k)
            mask_kj = is_kj_q & (
                is_prompt_k |
                kj_prev |
                (same_kj_block & (k >= ks_per_q) & (k <= q))
            )

            # last_j queries:
            same_lastj_block = is_lastj_q & is_lastj_k & (block_idx_q == block_idx_k)
            mask_lastj = is_lastj_q & (
                is_prompt_k |
                last_in_prev_last |
                (same_lastj_block & (k >= ls_per_q) & (k <= q))
            )

            return mask_prompt | mask_kj | mask_lastj

        block_mask = create_block_mask(
            mask_mod, B=1, H=heads, Q_LEN=L, KV_LEN=L, device="cuda", _compile=True,
        )
        return block_mask
    
    def _block_keep_mask_divergence_and_eos(
        input_ids: torch.Tensor,
        k_start: int,
        l_start: int,
        N: int,
        eos_id: int | None,
        drop_last_offset: bool = False,
    ) -> torch.Tensor:
        """
        Returns [N-1] bool if drop_last_offset else [N] bool.
        True => keep offset t for logits at position (start+t) predicting next token (t+1).
        We: start computing loss after first divergence, and stop at EOS.
        """
        device = input_ids.device
        size = N - 1 if drop_last_offset else N
        offs = torch.arange(size, device=device)

        k_block = input_ids[k_start : k_start + N]
        l_block = input_ids[l_start : l_start + N]

        # Divergence mask (pairwise): keep from first differing offset onward
        diff = (k_block[:size] != l_block[:size])
        if diff.any():
            first_diff = int(torch.nonzero(diff, as_tuple=False)[0])
            div_keep = offs >= first_diff
        else:
            div_keep = torch.zeros(size, dtype=torch.bool, device=device)

        # EOS mask: logits at offset t predict token t+1; if EOS is at offset e
        # in last_j, then offsets >= e are predicting post-EOS → mask them out
        _eos_ids = {eos_id} if eos_id is not None else set()
        _im_end_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
        if _im_end_id is not None:
            _eos_ids.add(_im_end_id)
        if _eos_ids:
            is_eos = torch.zeros(N, dtype=torch.bool, device=device)
            for eid in _eos_ids:
                is_eos |= (l_block == eid)
            eos_pos = is_eos.nonzero(as_tuple=False)
            if eos_pos.numel() > 0:
                first_eos = int(eos_pos[0])
                div_keep = div_keep & (offs < first_eos)

        return div_keep
    def _build_padding_mask_for_loss(input_ids: torch.Tensor, prompt_len: int, T: int) -> torch.Tensor:
        """
        [L] bool padding mask used for losses:
        True = mask, False = keep.
        Combines: (a) PAD tokens, (b) duplicate-prefix in each k_j vs last_j.
        """
        device = input_ids.device
        pad_id = "pad_token_id"

        mask = torch.zeros_like(input_ids, dtype=torch.bool, device=device)
        if pad_id is not None:
            mask |= (input_ids == pad_id)

        mask |= _duplicate_prefix_mask(input_ids, prompt_len, T)
        return mask
    def _duplicate_prefix_mask(input_ids: torch.Tensor, prompt_len: int, T: int) -> torch.Tensor:
        """
        [L] bool: True where token should be masked because it's in k_j's prefix
        identical to last_j (from left to right until first divergence).
        Only k_j tokens are masked; last_j tokens are not.
        """
        device = input_ids.device
        N = 32
        L = input_ids.size(0)
        mask = torch.zeros(L, dtype=torch.bool, device=device)

        k_starts, l_starts = _index_layout(prompt_len, T, N)
        for j in range(T):
            ks = k_starts[j]
            ls = l_starts[j]
            k_block = input_ids[ks:ks + N]
            l_block = input_ids[ls:ls + N]

            eq = (k_block == l_block)
            # find first index where they differ
            if torch.any(~eq):
                first_diff = int(torch.nonzero(~eq, as_tuple=False)[0])
            else:
                # fully identical: mask the whole k_j block
                first_diff = N

            if first_diff > 0:
                mask[ks:ks + first_diff] = True

        return mask
    def _one_pass_losses_step(model, inputs):
        
        input_ids, prompt_len, T = _unpack_sample(inputs)
        # for corner case of T=0, we still want to run a forward pass to avoid deadlocking on sync later
        if T == 0:
            outputs = model(
                input_ids=input_ids.unsqueeze(0),
                attention_mask=torch.ones_like(input_ids).unsqueeze(0),
                position_ids=torch.arange(input_ids.size(0), device=input_ids.device).unsqueeze(0),
            )
            logits = outputs.logits
            total_loss = logits.sum() * 0
            return total_loss.detach()
            
        # print(f"train_step={self.train_step_cnt}, prompt_len={prompt_len}, num_blocks={T}", flush=True)
        L = input_ids.size(0)

        eos_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id
        N = 32

        expected_len = prompt_len + 2 * T * N
        if L != expected_len:
            raise ValueError(
                f"Length mismatch: L={L}, expected {expected_len} "
                f"(prompt_len={prompt_len}, T={T}, n_token_sequence_size={N})"
            )

        # print(input_ids)

        attn_mask = torch.ones(L, dtype=torch.long, device=input_ids.device)
        k_starts, l_starts = _index_layout(prompt_len, T, N)

        # clean AR chain for potential ref usage / debugging: [prompt, last_0, last_1, ..., last_{T-1}]
        prompt_ids_block = input_ids[:prompt_len]
        l_blocks_concat = torch.cat([input_ids[ls:ls + N] for ls in l_starts], dim=0)
        ar_concat_ids = torch.cat([prompt_ids_block, l_blocks_concat], dim=0)

        # noisy chain for ref usage: [prompt, k_0, k_1, ..., k_{T-1}]
        k_blocks_concat = torch.cat([input_ids[ks:ks + N] for ks in k_starts], dim=0)
        noisy_concat_ids = torch.cat([prompt_ids_block, k_blocks_concat], dim=0)

        # ===== mutate post-EOS tokens in the final last block to PAD as before =====
        _flip_block_after_eos_to_pad(input_ids, l_starts[-1], N, eos_id, pad_id)

        # ===== block-structured forward for both modes =====
        num_heads = 28
        position_ids = _build_shared_position_ids(L, prompt_len, T)

        # position mapping: k_j and last_j positions in the interleaved sequence
        offs_full = torch.arange(N, device="cuda")
        blk_k_pos = []    # noisy k_j positions
        blk_last_pos = []  # clean last_j positions
        block_j_idx = []   # which block j each position belongs to
        intra_idx = []     # offset within block
        for j in range(T):
            blk_k_pos.append(k_starts[j] + offs_full)
            blk_last_pos.append(l_starts[j] + offs_full)
            block_j_idx.append(torch.full((N,), j, device="cuda", dtype=torch.long))
            intra_idx.append(offs_full.clone())
        blk_k_pos = torch.cat(blk_k_pos, dim=0)      # [T*N]
        blk_last_pos = torch.cat(blk_last_pos, dim=0)  # [T*N]
        block_j_idx = torch.cat(block_j_idx, dim=0)
        intra_idx = torch.cat(intra_idx, dim=0)

        # identify noised positions: from first divergence onward in each block,
        # excluding last offset (no next token within block as golden)
        keep_parts = []
        for j in range(T):
            div_mask = _block_keep_mask_divergence_and_eos(
                input_ids, k_starts[j], l_starts[j], N, eos_id=eos_id, drop_last_offset=True,
            )  # [N-1] bool: True from first diff onward, excludes last offset
            # pad to N with False for offset N-1
            padded = F.pad(div_mask, (0, 1), value=False)
            kept_offsets = padded.nonzero(as_tuple=False).squeeze(-1).tolist()
            print(f"  [consistency keep] block j={j}: kept offsets = {kept_offsets} ({len(kept_offsets)}/{N})")
            keep_parts.append(padded)
        keep = torch.cat(keep_parts, dim=0)  # [T*N]

        # filter down to noised positions only
        keep_idx = keep.nonzero(as_tuple=False).squeeze(-1)
        blk_k_pos_f = blk_k_pos[keep_idx]
        blk_last_pos_f = blk_last_pos[keep_idx]
        block_j_idx_f = block_j_idx[keep_idx]
        intra_idx_f = intra_idx[keep_idx]

        # golden = next token in the clean (last_j) block: input_ids[last_pos + 1]
        golden_ids = input_ids[blk_last_pos_f + 1]  # [M]

        # collect logits from both modes
        mode_logits = {}
        for mode in ("same", "cross"):
            blk_mask = _build_block_mask(L, prompt_len, T, num_heads, mode=mode)
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids.unsqueeze(0),
                    attention_mask=blk_mask,
                    position_ids=position_ids.unsqueeze(0),
                )
            mode_logits[mode] = outputs.logits[0]  # [L, V]

        # ===== ref model prefill on clean AR chain =====
        ref_ar_pos = torch.tensor(
            [prompt_len + int(block_j_idx_f[i]) * N + int(intra_idx_f[i])
             for i in range(blk_k_pos_f.size(0))],
            device="cuda", dtype=torch.long,
        )
        with torch.no_grad():
            ref_outputs = ref_model(
                input_ids=ar_concat_ids.unsqueeze(0).to("cuda:1"),
                attention_mask=torch.ones(1, ar_concat_ids.size(0), device="cuda:1", dtype=torch.long),
                use_cache=False,
            )
            ref_logits_full = ref_outputs.logits[0].to("cuda")  # [prompt_len + T*N, V]

        # ===== precompute log-probs and top-K for both modes + clean =====
        K = 10
        M = blk_k_pos_f.size(0)

        clean_log_p = F.log_softmax(mode_logits["same"][blk_last_pos_f].float(), dim=-1)
        clean_topk_vals, clean_topk_ids = clean_log_p.topk(K, dim=-1)

        ref_log_p = F.log_softmax(ref_logits_full[ref_ar_pos].float(), dim=-1)
        ref_topk_vals, ref_topk_ids = ref_log_p.topk(K, dim=-1)

        mode_data = {}
        for mode in ("same", "cross"):
            logits_full = mode_logits[mode]
            noisy_log_p = F.log_softmax(logits_full[blk_k_pos_f].float(), dim=-1)
            topk_vals, topk_ids = noisy_log_p.topk(K, dim=-1)
            kl = F.kl_div(noisy_log_p, clean_log_p, log_target=True, reduction="batchmean")
            mode_data[mode] = {
                "noisy_log_p": noisy_log_p,
                "topk_vals": topk_vals,
                "topk_ids": topk_ids,
                "kl": kl,
            }

        # ===== print summary KL =====
        kl_ref = F.kl_div(ref_log_p, clean_log_p, log_target=True, reduction="batchmean")
        print(f"[mode=ref  ] KL(clean || ref)   = {kl_ref.item():.6f}  ({M} noised positions)")
        for mode in ("same", "cross"):
            print(f"[mode={mode:5s}] KL(clean || noisy) = {mode_data[mode]['kl'].item():.6f}  ({M} noised positions)")

        # ===== print per-position: clean, ref, same, cross =====
        for i in range(M):
            j = int(block_j_idx_f[i])
            off = int(intra_idx_f[i])
            golden_tok = int(golden_ids[i])
            golden_str = tokenizer.decode([golden_tok])
            noisy_input_tok = int(input_ids[blk_k_pos_f[i]])
            noisy_input_str = tokenizer.decode([noisy_input_tok])
            clean_input_tok = int(input_ids[blk_last_pos_f[i]])
            clean_input_str = tokenizer.decode([clean_input_tok])

            print(f"\n{'='*80}")
            print(f"block j={j}, offset={off} | "
                  f"golden_next: {repr(golden_str)} | "
                  f"input_clean: {repr(clean_input_str)} | input_noisy: {repr(noisy_input_str)}")

            golden_clean_lp = float(clean_log_p[i, golden_tok])
            clean_toks = [f"{repr(tokenizer.decode([int(clean_topk_ids[i,k])]))} ({float(clean_topk_vals[i,k]):.4f})"
                          for k in range(K)]
            print(f"  [clean] golden_lp={golden_clean_lp:.4f} | top{K}: {' | '.join(clean_toks)}")

            golden_ref_lp = float(ref_log_p[i, golden_tok])
            ref_toks = [f"{repr(tokenizer.decode([int(ref_topk_ids[i,k])]))} ({float(ref_topk_vals[i,k]):.4f})"
                        for k in range(K)]
            print(f"  [ref  ] golden_lp={golden_ref_lp:.4f} | top{K}: {' | '.join(ref_toks)}")

            for mode in ("same", "cross"):
                d = mode_data[mode]
                golden_lp = float(d["noisy_log_p"][i, golden_tok])
                toks = [f"{repr(tokenizer.decode([int(d['topk_ids'][i,k])]))} ({float(d['topk_vals'][i,k]):.4f})"
                        for k in range(K)]
                print(f"  [{mode:5s}] golden_lp={golden_lp:.4f} | top{K}: {' | '.join(toks)}")

        # =================================================================
        # ===== Compute loss_consistency + loss_ref with backward =====
        # =================================================================
        T_soft = 1.0
        tau_ref = 1.0
        padding_mask_consistency = torch.zeros(M, dtype=torch.bool, device="cuda")

        # ref loss shared setup: ALL last_j positions
        offs_full_all = torch.arange(N, device="cuda")
        ref_student_pos_list = []
        ref_teacher_pos_list = []
        for j in range(T):
            ls = l_starts[j]
            clean_base = prompt_len + j * N
            ref_student_pos_list.append(ls + offs_full_all)
            ref_teacher_pos_list.append(clean_base + offs_full_all)
        ref_student_pos = torch.cat(ref_student_pos_list, dim=0)  # [T*N]
        ref_teacher_pos = torch.cat(ref_teacher_pos_list, dim=0)  # [T*N]
        ref_teacher_logits = ref_logits_full.index_select(0, ref_teacher_pos)  # [T*N, V]
        ref_targets = ref_teacher_logits.argmax(dim=-1)  # [T*N]

        # Mask out positions after EOS / <|im_end|> in each last_j block
        eos_ids = {tokenizer.eos_token_id}
        im_end_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
        if im_end_id is not None and im_end_id != tokenizer.unk_token_id:
            eos_ids.add(im_end_id)
        for j in range(T):
            ls = l_starts[j]
            block = input_ids[ls:ls + N]
            # find first occurrence of any EOS-like token
            is_eos = torch.zeros(N, dtype=torch.bool, device=block.device)
            for eid in eos_ids:
                is_eos |= (block == eid)
            eos_pos = is_eos.nonzero(as_tuple=False)
            if eos_pos.numel() > 0:
                first_eos = int(eos_pos[0])
                # mask from first_eos (inclusive): logits at first_eos predict post-EOS token
                start_idx = j * N + first_eos
                end_idx = (j + 1) * N
                print(f"  [EOS mask] block j={j}: EOS at off={first_eos}, masking ref_targets[{start_idx}:{end_idx}]")
                if start_idx < end_idx:
                    ref_targets[start_idx:end_idx] = -100
        

        results = {}  # mode -> {loss_consistency, loss_ref, grad_norm_consistency, ...}

        for attn_mode in ("same", "cross"):
            print(f"\n\n{'#'*80}")
            print(f"# LOSS & GRADIENT ANALYSIS (mode={attn_mode})")
            print(f"{'#'*80}")

            blk_mask = _build_block_mask(L, prompt_len, T, num_heads, mode=attn_mode)

            # ----- Forward #1: consistency loss -----
            model.zero_grad()
            outputs_c = model(
                input_ids=input_ids.unsqueeze(0),
                attention_mask=blk_mask,
                position_ids=position_ids.unsqueeze(0),
            )
            logits_c = outputs_c.logits

            student_logits = logits_c[0, blk_k_pos_f, :] / T_soft
            teacher_logits = logits_c[0, blk_last_pos_f, :].detach() / T_soft

            loss_consistency = soft_cross_entropy(
                student_logits.float(),
                teacher_logits.float(),
                padding_mask_consistency,
            ) * (T_soft * T_soft) / T

            print(f"\nloss_consistency = {loss_consistency.item():.6f}")

            loss_consistency.backward()
            grad_norm_consistency = 0.0
            grad_norms_per_layer_consistency = {}
            for name, p in model.named_parameters():
                if p.grad is not None:
                    pnorm = p.grad.float().norm().item()
                    grad_norm_consistency += pnorm ** 2
                    grad_norms_per_layer_consistency[name] = pnorm
            grad_norm_consistency = grad_norm_consistency ** 0.5
            print(f"  grad_norm(consistency) = {grad_norm_consistency:.6f}")

            sorted_layers = sorted(grad_norms_per_layer_consistency.items(), key=lambda x: -x[1])[:5]
            for lname, lnorm in sorted_layers:
                print(f"    {lname}: {lnorm:.6f}")

            # ----- Forward #2: ref loss -----
            model.zero_grad()
            outputs_r = model(
                input_ids=input_ids.unsqueeze(0),
                attention_mask=blk_mask,
                position_ids=position_ids.unsqueeze(0),
            )
            logits_r = outputs_r.logits

            ref_student_logits = logits_r[0, ref_student_pos, :]

            # ----- DEBUG: per-position log-prob of ref target under student -----
            with torch.no_grad():
                student_lp = F.log_softmax(ref_student_logits.float() / tau_ref, dim=-1)  # [T*N, V]
                valid_mask = ref_targets != -100
                student_argmax = ref_student_logits.argmax(dim=-1)

                # only compute stats on valid (non-masked) positions
                valid_targets = ref_targets[valid_mask]
                valid_lp = student_lp[valid_mask]
                target_lp = valid_lp[torch.arange(valid_targets.size(0), device="cuda"), valid_targets]
                valid_argmax = student_argmax[valid_mask]
                agree = (valid_argmax == valid_targets).sum().item()
                total_valid = valid_targets.size(0)
                total_all = ref_targets.size(0)

                print(f"\n  [ref debug] valid positions: {total_valid}/{total_all} (masked {total_all - total_valid} post-EOS)")
                print(f"  [ref debug] student log-prob at ref_target: "
                      f"mean={target_lp.mean().item():.4f}  min={target_lp.min().item():.4f}  max={target_lp.max().item():.4f}")
                print(f"  [ref debug] argmax agreement: {agree}/{total_valid} ({100*agree/total_valid:.1f}%)")
                print(f"  [ref debug] per-position target_lp:")
                for idx in range(total_all):
                    j_idx = idx // N
                    off = idx % N
                    if int(ref_targets[idx]) == -100:
                        print(f"    j={j_idx} off={off:2d} | [MASKED post-EOS]")
                        continue
                    ref_tok = int(ref_targets[idx])
                    stu_tok = int(student_argmax[idx])
                    lp_val = float(student_lp[idx, ref_tok])
                    print(f"    j={j_idx} off={off:2d} | ref_target={repr(tokenizer.decode([ref_tok])):12s} "
                          f"student_argmax={repr(tokenizer.decode([stu_tok])):12s} | student_lp={lp_val:.4f}"
                          f"{'  <<<' if ref_tok != stu_tok else ''}")

            loss_ref = F.cross_entropy(
                ref_student_logits.float() / tau_ref,
                ref_targets,
                reduction="mean",
                ignore_index=-100,
            ) * 10

            print(f"\nloss_ref = {loss_ref.item():.6f}")

            loss_ref.backward()
            grad_norm_ref = 0.0
            grad_norms_per_layer_ref = {}
            for name, p in model.named_parameters():
                if p.grad is not None:
                    pnorm = p.grad.float().norm().item()
                    grad_norm_ref += pnorm ** 2
                    grad_norms_per_layer_ref[name] = pnorm
            grad_norm_ref = grad_norm_ref ** 0.5
            print(f"  grad_norm(ref) = {grad_norm_ref:.6f}")

            sorted_layers_ref = sorted(grad_norms_per_layer_ref.items(), key=lambda x: -x[1])[:5]
            for lname, lnorm in sorted_layers_ref:
                print(f"    {lname}: {lnorm:.6f}")

            results[attn_mode] = {
                "loss_consistency": loss_consistency.item(),
                "loss_ref": loss_ref.item(),
                "grad_norm_consistency": grad_norm_consistency,
                "grad_norm_ref": grad_norm_ref,
            }

        # ----- combined summary -----
        print(f"\n\n{'='*80}")
        print(f"SUMMARY (M={M} noised positions, T*N={T*N} ref positions):")
        for attn_mode in ("same", "cross"):
            r = results[attn_mode]
            print(f"  [{attn_mode:5s}] loss_consistency={r['loss_consistency']:.6f}  grad={r['grad_norm_consistency']:.6f}"
                  f"  |  loss_ref={r['loss_ref']:.6f}  grad={r['grad_norm_ref']:.6f}"
                  f"  |  ratio={r['grad_norm_ref'] / (r['grad_norm_consistency'] + 1e-12):.4f}")

        model.zero_grad()
        return
        #         with torch.no_grad():
        #             ref_param = next(self.ref_model.parameters())
        #             ref_device = ref_param.device

        #             ref_input_ids = ar_concat_ids.to(ref_device).unsqueeze(0)
        #             ref_attn = torch.ones_like(ar_concat_ids, device=ref_device).unsqueeze(0)

        #             ref_outputs = self.ref_model(
        #                 input_ids=ref_input_ids,
        #                 attention_mask=ref_attn,
        #                 use_cache=False,
        #             )
        #             ref_logits_full = ref_outputs.logits[0].to(input_ids.device)   # [prompt_len + T*N, V]

        #         ref_logits_all = ref_logits_full.index_select(0, ref_teacher_pos)  # [T*N, V]

        #         tau_ref = getattr(self.args, "ref_temperature", 1.0)

        #         ref_targets = ref_logits_all.argmax(dim=-1)   # [T*N]

        #         # ===== DEBUG PRINTING =====
        #         if self.args.local_rank == 0:
        #             print(
        #                 f"===== decoded last_N ref targets =====\n"
        #                 f"{self.processing_class.decode(ref_targets[-64:])}\n==========\n",
        #                 flush=True,
        #             )
        #             print(
        #                 f"===== last_N ref tokens =====\n{ref_targets[-64:]}\n==========\n",
        #                 flush=True,
        #             )
        #         # ===== DEBUG PRINTING =====                

        #         loss_ref = F.cross_entropy(
        #             ref_student_logits.float() / tau_ref,
        #             ref_targets,
        #             reduction="mean",
        #         )
        #     else:
        #         loss_ref = torch.zeros((), device=self.args.device)

        # # =========================================================
        # # Total loss switch
        # #   with ref_model: consistency + ref loss
        # #   without ref_model: consistency + ar loss
        # # =========================================================
        # if self.ref_model is not None:
        #     ref_weight = getattr(self.args, "ref_weight", 1.0)
        #     if ref_weight == 0.0:
        #         print("[warning] ref_weight=0.0, consistency loss only", flush=True)
        #         total_loss = loss_consistency
        #     else:
        #         print(f"[info] ref_weight={ref_weight}, combining consistency and ref losses", flush=True)
        #         total_loss = loss_consistency + ref_weight * loss_ref
        # else:
        #     print("[info] no ref model, using AR loss as auxiliary to consistency loss", flush=True)
        #     total_loss = loss_consistency + loss_ar

        # if self.args.qlora:
        #     total_loss.requires_grad = True

        # if self.args.local_rank == 0:
        #     log_dict = {
        #         "consistency loss": float(loss_consistency.detach().cpu()),
        #         "ar loss": float(loss_ar.detach().cpu()),
        #         "ref loss": float(loss_ref.detach().cpu()),
        #         "total loss": float(total_loss.detach().cpu()),
        #     }
        #     if self.ref_model is not None:
        #         log_dict["ref loss"] = float(loss_ref.detach().cpu())
        #     wandb.log(log_dict)

        # torch.cuda.empty_cache()

        # with self.accelerator.accumulate(model):
        #     self.accelerator.backward(total_loss)

        # return total_loss.detach()
    
    _one_pass_losses_step(model, output)