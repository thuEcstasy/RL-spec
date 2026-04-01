import torch
import wandb
from torch.cuda.amp import autocast
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother

import torch.nn.functional as F

from torch.nn.attention.flex_attention import create_block_mask

from functools import lru_cache

import torch.distributed as dist
import deepspeed

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

class CllmTrainer(Trainer):
    def __init__(self, *args,  accelerator=None, optimizer=None, lr_scheduler=None, train_dataloader=None, ref_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        args = kwargs["args"]

        self.accelerator = accelerator
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader

        self.base_model = self.accelerator.unwrap_model(self.model)
        
        # ref model
        self.ref_model = ref_model
        if self.ref_model is not None:
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad_(False)
        
        self.cfg = self.base_model.config

        self.train_step_cnt = 0

        self.max_new_tokens = args.max_new_tokens
        self.use_gt_labels = args.use_gt_labels
        # cache BlockMasks keyed by (L, prompt_len, T, heads, version)
        self._blockmask_cache = {}

    # Utilities
    @staticmethod
    def _to_int(x):
        return x.item() if isinstance(x, torch.Tensor) else int(x)

    def get_train_dataloader(self):
        return self.train_dataloader

    def _unpack_sample(self, inputs):
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
        prompt_len = self._to_int(prompt_len)

        traj_position_indices = inputs["traj_position_indices"][0]
        traj_position_indices = [int(u) for u in traj_position_indices]
        T = len(traj_position_indices)

        return (
            input_ids.to(self.args.device),
            prompt_len,
            T,
        )
    
    def _duplicate_prefix_mask(self, input_ids: torch.Tensor, prompt_len: int, T: int) -> torch.Tensor:
        """
        [L] bool: True where token should be masked because it's in k_j's prefix
        identical to last_j (from left to right until first divergence).
        Only k_j tokens are masked; last_j tokens are not.
        """
        device = input_ids.device
        N = self.max_new_tokens
        L = input_ids.size(0)
        mask = torch.zeros(L, dtype=torch.bool, device=device)

        k_starts, l_starts = self._index_layout(prompt_len, T, N)
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

    def _build_padding_mask_for_loss(self, input_ids: torch.Tensor, prompt_len: int, T: int) -> torch.Tensor:
        """
        [L] bool padding mask used for losses:
        True = mask, False = keep.
        Combines: (a) PAD tokens, (b) duplicate-prefix in each k_j vs last_j.
        """
        device = input_ids.device
        pad_id = getattr(self.processing_class, "pad_token_id", None)

        mask = torch.zeros_like(input_ids, dtype=torch.bool, device=device)
        if pad_id is not None:
            mask |= (input_ids == pad_id)

        mask |= self._duplicate_prefix_mask(input_ids, prompt_len, T)
        return mask

    def _block_keep_mask_divergence_and_eos(
        self,
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
        We: start computing loss after first divergence.
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

        # UNUSED — EOS mask: keep offsets t such that (t+1) is BEFORE EOS in both blocks
        #def keep_next_before_eos(block):
        #    if eos_id is None:
        #        return torch.ones(size, dtype=torch.bool, device=device)
        #    pos = torch.nonzero(block == eos_id, as_tuple=False)
        #    if pos.numel() == 0:
        #        return torch.ones(size, dtype=torch.bool, device=device)
        #    e = int(pos[0])          # EOS index within [0..N-1]
        #    return offs < e

        #eos_keep = keep_next_before_eos(k_block) & keep_next_before_eos(l_block)

        return div_keep

    @staticmethod
    def _index_layout(prompt_len: int, T: int, N):
        """Return lists of start indices for all k_j and last_j blocks in flattened sequence."""
        k_starts = [prompt_len + 2 * j * N for j in range(T)]
        l_starts = [prompt_len + (2 * j + 1) * N for j in range(T)]
        return k_starts, l_starts

    def _build_shared_position_ids(self, L: int, prompt_len: int, T: int):
        """
        Build [L] position_ids so each (k_j, last_j) share the same positions.
        Prompt: 0...prompt_len-1
        For each j: both blocks use prompt_len + j*N .. prompt_len + j*N + (N-1)
        """
        device = self.args.device
        N = self.max_new_tokens
        pos = torch.empty(L, dtype=torch.long, device=device)

        # Prompt positions
        pos[:prompt_len] = torch.arange(prompt_len, device=device)

        # Pair-shared positions
        k_starts, l_starts = self._index_layout(prompt_len, T, N)
        rel = torch.arange(N, device=device)
        for j in range(T):
            base = prompt_len + j * N
            ks = k_starts[j]
            ls = l_starts[j]
            pos[ks:ks + N] = base + rel
            pos[ls:ls + N] = base + rel

        return pos
    
    def _flip_block_after_eos_to_pad(self, input_ids: torch.Tensor, start: int, N: int, eos_id: int | None, pad_id: int | None) -> int:
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
    
    def soft_cross_entropy(self, predicts, targets, padding_mask):
        if (~padding_mask).sum() == 0:
            return 0 * predicts[0][0]
        predict_log_prob = torch.nn.functional.log_softmax(predicts, dim=-1)
        targets_prob = torch.nn.functional.softmax(targets, dim=-1)
        entropy = -targets_prob * predict_log_prob
        expand_mask = padding_mask.unsqueeze(-1).expand_as(entropy)
        entropy = entropy.masked_fill(expand_mask, 0)
        mean_entropy = entropy.sum() / (~padding_mask).sum()
        return mean_entropy

    # FlexAttention BlockMask
    # - prompt queries: causal within prompt
    # - k_j queries: causal within *their own* k_j block + prompt
    # - last_j queries: causal within *their own* last_j block + prompt
    def _build_block_mask(self, L: int, prompt_len: int, T: int, heads: int):
        N = self.max_new_tokens
        k_starts, l_starts = self._index_layout(prompt_len, T, N)

        ks = torch.tensor(k_starts, device=self.args.device)
        ls = torch.tensor(l_starts, device=self.args.device)

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

            # for k_j queries, allow attending to ALL PREVIOUS k_* blocks
            # (k blocks have even block indices: 0, 2, 4, ... 2*(j_q-1))
            k_in_prev_k   = is_kj_k    & (block_idx_k < 2 * j_q)
            # keep old behavior for last_j (still sees previous last_*)
            last_in_prev_last = is_lastj_k & (block_idx_k < 2 * j_q)

            # prompt is always causal
            mask_prompt = is_prompt_q & (k <= q)

            # k_j queries:
            same_kj_block = is_kj_q & is_kj_k & (block_idx_q == block_idx_k)
            mask_kj = is_kj_q & (
                is_prompt_k |
                k_in_prev_k |  # attends to "previous k_*"
                (same_kj_block & (k >= ks_per_q) & (k <= q))
            )

            # last_j queries (UNCHANGED):
            same_lastj_block = is_lastj_q & is_lastj_k & (block_idx_q == block_idx_k)
            mask_lastj = is_lastj_q & (
                is_prompt_k |
                last_in_prev_last |  # attends to all previous last_{<j}
                (same_lastj_block & (k >= ls_per_q) & (k <= q))
            )

            return mask_prompt | mask_kj | mask_lastj

        block_mask = create_block_mask(
            mask_mod, B=1, H=heads, Q_LEN=L, KV_LEN=L, device=self.args.device, _compile=True,
        )
        return block_mask

    @torch.no_grad()
    def maybe_sync_rollout_model(self):
        if not hasattr(self, "rollout_dataset"):
            return

        ds = self.rollout_dataset
        if ds is None or getattr(ds, "rollout_model", None) is None:
            return

        every = getattr(self.args, "rollout_sync_every", 1)
        if self.train_step_cnt % every != 0:
            return

        train_base = self.accelerator.unwrap_model(self.model)
        rollout_model = ds.rollout_model
        rollout_model.eval()

        # detect zero stage if available
        zero_stage = None
        if hasattr(self.accelerator.state, "deepspeed_plugin") and self.accelerator.state.deepspeed_plugin is not None:
            try:
                zero_stage = self.accelerator.state.deepspeed_plugin.zero_stage
            except Exception:
                zero_stage = None

        # ZeRO-1 / ZeRO-2: state_dict usually works
        if zero_stage in (None, 0, 1, 2):
            src_sd = train_base.state_dict()
            missing, unexpected = rollout_model.load_state_dict(src_sd, strict=False)
            if self.accelerator.is_main_process and (missing or unexpected):
                print(f"[sync] missing={missing[:5]} unexpected={unexpected[:5]}", flush=True)
            rollout_model.eval()
            return

        # ZeRO-3: gather one parameter at a time
        src_named_params = dict(train_base.named_parameters())
        dst_named_params = dict(rollout_model.named_parameters())

        # sanity check on names
        common_param_names = [n for n in src_named_params.keys() if n in dst_named_params]

        for name in common_param_names:
            src_p = src_named_params[name]
            dst_p = dst_named_params[name]

            # Gather full parameter temporarily on each rank
            with deepspeed.zero.GatheredParameters([src_p], modifier_rank=None):
                if src_p.data.numel() == 0:
                    raise RuntimeError(f"[sync] gathered param still empty: {name}, shape={tuple(src_p.shape)}")

                if src_p.shape != dst_p.shape:
                    raise RuntimeError(
                        f"[sync] shape mismatch for {name}: "
                        f"src={tuple(src_p.shape)} dst={tuple(dst_p.shape)}"
                    )

                dst_p.data.copy_(src_p.data.to(device=dst_p.device, dtype=dst_p.dtype))

        # buffers are not partitioned the same way; just copy directly
        src_named_bufs = dict(train_base.named_buffers())
        dst_named_bufs = dict(rollout_model.named_buffers())
        common_buf_names = [n for n in src_named_bufs.keys() if n in dst_named_bufs]

        for name in common_buf_names:
            src_b = src_named_bufs[name]
            dst_b = dst_named_bufs[name]
            if src_b.shape != dst_b.shape:
                raise RuntimeError(
                    f"[sync] buffer shape mismatch for {name}: "
                    f"src={tuple(src_b.shape)} dst={tuple(dst_b.shape)}"
                )
            dst_b.data.copy_(src_b.data.to(device=dst_b.device, dtype=dst_b.dtype))

        rollout_model.eval()

    def training_step(self, model, inputs, num_items_in_batch=None):
        # [B, L]
        bsz = inputs["prompt_ids"].size(0)
        losses = []
        
        for bidx in range(bsz):
            while output is None:
                output = self.rollout_dataset._build_training_sample(inputs["prompt_ids"][bidx])
            losses.append(self._one_pass_losses_step(model, output))

        self.train_step_cnt += 1
        
        did_sync = False
        if self.train_step_cnt == 1 or self.train_step_cnt % self.args.rollout_sync_every == 0:
            self.maybe_sync_rollout_model()
            did_sync = True
            
        if did_sync and torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        if output is not None and "tokens_per_iter" in output:
            tpi_local = torch.tensor(
                [output["tokens_per_iter"]],
                dtype=torch.float,
                device=self.args.device,
            )
        else:
            tpi_local = torch.tensor([0.0], dtype=torch.float, device=self.args.device)

        if hasattr(self, "accelerator"):
            tpi_all = self.accelerator.gather(tpi_local)
            tpi_mean = tpi_all.mean().item()
        else:
            tpi_mean = tpi_local.item()

        return torch.stack(losses).mean()

    def _one_pass_losses_step(self, model, inputs):
        input_ids, prompt_len, T = self._unpack_sample(inputs)
        # print(f"train_step={self.train_step_cnt}, prompt_len={prompt_len}, num_blocks={T}", flush=True)
        L = input_ids.size(0)

        eos_id = getattr(self.processing_class, "eos_token_id")
        pad_id = getattr(self.processing_class, "pad_token_id")
        N = self.max_new_tokens

        expected_len = prompt_len + 2 * T * N
        if L != expected_len:
            raise ValueError(
                f"Length mismatch: L={L}, expected {expected_len} "
                f"(prompt_len={prompt_len}, T={T}, n_token_sequence_size={N})"
            )

        attn_mask = torch.ones(L, dtype=torch.long, device=input_ids.device)
        k_starts, l_starts = self._index_layout(prompt_len, T, N)

        # clean AR chain for potential ref usage / debugging: [prompt, last_0, last_1, ..., last_{T-1}]
        prompt_ids_block = input_ids[:prompt_len]
        l_blocks_concat = torch.cat([input_ids[ls:ls + N] for ls in l_starts], dim=0)
        ar_concat_ids = torch.cat([prompt_ids_block, l_blocks_concat], dim=0)

        # noisy chain for ref usage: [prompt, k_0, k_1, ..., k_{T-1}]
        k_blocks_concat = torch.cat([input_ids[ks:ks + N] for ks in k_starts], dim=0)
        noisy_concat_ids = torch.cat([prompt_ids_block, k_blocks_concat], dim=0)

        # ===== mutate post-EOS tokens in the final last block to PAD as before =====
        self._flip_block_after_eos_to_pad(input_ids, l_starts[-1], N, eos_id, pad_id)

        # ===== block-structured student forward =====
        num_heads = getattr(self.cfg, "num_attention_heads", 28)
        blk_mask = self._build_block_mask(L, prompt_len, T, num_heads)
        position_ids = self._build_shared_position_ids(L, prompt_len, T)


        outputs = model(
            input_ids=input_ids.unsqueeze(0),
            attention_mask=blk_mask,
            position_ids=position_ids.unsqueeze(0),
            attn_implementation="flex_attention",
        )
        logits = outputs.logits  # [1, L, V]

        # =========================================================
        # AR loss (原来的计算部分保留)
        # =========================================================
        pair_logit_positions = []
        pair_target_positions = []

        def add_forward_pairs(seg_start: int, seg_end: int):
            """
            Add all in-block next-token pairs for a segment [seg_start, seg_end).
            Produces pairs (p --> p+1) for p in [seg_start .. seg_end-2].
            """
            p = torch.arange(seg_start, seg_end - 1, device=self.args.device, dtype=torch.long)
            t = p + 1
            pair_logit_positions.append(p)
            pair_target_positions.append(t)

        # Prompt segment
        end_prompt = prompt_len
        add_forward_pairs(0, end_prompt)

        # for each j, compute first-token bridge + in-block AR pairs on last_j
        for j in range(T):
            ls = l_starts[j]

            # bridge token
            if j == 0:
                logit_pos = end_prompt - 1
                target_pos = ls
            else:
                prev_ls = l_starts[j - 1]
                logit_pos = prev_ls + (N - 1)
                target_pos = ls

            pair_logit_positions.append(torch.tensor([logit_pos], device=self.args.device))
            pair_target_positions.append(torch.tensor([target_pos], device=self.args.device))

            block = input_ids[ls: ls + N]

            eos_pos = None
            if eos_id is not None:
                epos = torch.nonzero(block == eos_id, as_tuple=False)
                eos_pos = int(epos[0]) if epos.numel() > 0 else None

            end = N
            if eos_pos is not None:
                end = min(end, eos_pos + 1)  # include EOS in segment

                # mark PAD as 0 for attn_mask
                if pad_id is not None:
                    mask_block = attn_mask[ls: ls + N]
                    mask_block[block == pad_id] = 0
                    attn_mask[ls: ls + N] = mask_block

            add_forward_pairs(ls, ls + end)

        # Compute CE over all AR pairs (保留原逻辑)
        if len(pair_logit_positions) == 0:
            loss_ar = torch.zeros((), device=self.args.device)
        else:
            p_all = torch.cat(pair_logit_positions, dim=0)
            t_all = torch.cat(pair_target_positions, dim=0)

            ar_logits = logits[0, p_all, :].clone()                         # [K, V]
            ar_targets = input_ids.index_select(0, t_all).clone().detach() # [K]

            # ===== DEBUG PRINTING =====
            if self.args.local_rank == 0:
                print(
                    f"===== decoded last_N AR targets =====\n"
                    f"{self.processing_class.decode(ar_targets[-64:])}\n==========\n",
                    flush=True,
                )
                print(
                    f"===== last_N AR tokens =====\n{ar_targets[-64:]}\n==========\n",
                    flush=True,
                )
            # ===== DEBUG PRINTING =====

            if pad_id is not None:
                ar_targets[ar_targets == pad_id] = -100


            loss_ar = F.cross_entropy(
                ar_logits.float(),
                ar_targets,
                reduction="mean",
                label_smoothing=0.0,
                ignore_index=-100,
            ) * 10

        # =========================================================
        # Consistency loss (原逻辑保留)
        # =========================================================
        T_soft = getattr(self.args, "distill_temperature", 1.0)

        drop_last_offset = False
        offs = torch.arange(N - 1 if drop_last_offset else N, device=self.args.device)

        student_positions, teacher_positions = [], []
        for j in range(T):
            ks, ls = k_starts[j], l_starts[j]
            pair_keep = self._block_keep_mask_divergence_and_eos(
                input_ids, ks, ls, N, eos_id=eos_id, drop_last_offset=drop_last_offset
            )
            if pair_keep.any():
                sp = ks + offs[pair_keep]   # noisy k_j positions
                tp = ls + offs[pair_keep]   # clean last_j positions
                student_positions.append(sp)
                teacher_positions.append(tp)


        if len(student_positions) == 0:
            zero = logits.sum() * 0.0
            loss_consistency = zero
            loss_ref = zero
        else:
            sp = torch.cat(student_positions, dim=0)  # [K]
            tp = torch.cat(teacher_positions, dim=0)  # [K]

            # global [L] padding mask: PADs and duplicate k_j prefixes
            global_pad_and_dup_mask = self._build_padding_mask_for_loss(input_ids, prompt_len, T)

            # per-pair padding mask (True = mask out)
            padding_mask = global_pad_and_dup_mask.index_select(0, sp)

            # ---------- student noisy logits ----------
            student_logits_all = logits[0, sp, :]   # [K, V]

            # ---------- original self-distill consistency ----------
            teacher_logits_all = logits[0, tp, :].detach()

            student_logits_temp = student_logits_all / T_soft
            teacher_logits_temp = teacher_logits_all / T_soft

            # [DEBUG] print some of their probs to check for collapse
            # if self.args.local_rank == 0:
            #     with torch.no_grad():
            #         print(
            #             f"===== sample teacher probs for first 5 pairs =====\n"
            #             f"{teacher_logits_temp[:16][:100]}\n==========\n",
            #             flush=True,
            #         )
            #         print(
            #             f"===== sample student probs for first 5 pairs =====\n"
            #             f"{student_logits_temp[:16][:100]}\n==========\n",
            #             flush=True,
            #         )
            #         print(logits[0, prompt_len + 1, :10])
            #         print(logits[0, prompt_len + N + 1, :10])

            loss_consistency = self.soft_cross_entropy(
                student_logits_temp.float(),
                teacher_logits_temp.float(),
                padding_mask
            )
            loss_consistency = loss_consistency * (T_soft * T_soft) / T

            # =========================================================
            # New ref loss: KL(student noisy logits || ref logits on same noisy chain)
            # 实现上用 kl_div(log student, prob teacher) = KL(teacher || student)
            # =========================================================
            if self.ref_model is not None:
                # map original positions in k_j blocks -> positions in noisy_concat_ids
                orig2noisy = torch.full((L,), -1, dtype=torch.long, device=input_ids.device)
                orig2noisy[:prompt_len] = torch.arange(prompt_len, device=input_ids.device)

                cursor = prompt_len
                for ks in k_starts:
                    orig2noisy[ks:ks + N] = torch.arange(cursor, cursor + N, device=input_ids.device)
                    cursor += N

                ref_pos = orig2noisy.index_select(0, sp)
                if (ref_pos < 0).any():
                    bad = (ref_pos < 0).nonzero(as_tuple=False).view(-1)[:8]
                    raise RuntimeError(f"Found unmapped noisy positions for ref model: {bad.tolist()}")

                # run ref model on noisy chain
                with torch.no_grad():
                    ref_param = next(self.ref_model.parameters())
                    ref_device = ref_param.device

                    ref_input_ids = noisy_concat_ids.to(ref_device).unsqueeze(0)
                    ref_attn = torch.ones_like(noisy_concat_ids, device=ref_device).unsqueeze(0)

                    ref_outputs = self.ref_model(
                        input_ids=ref_input_ids,
                        attention_mask=ref_attn,
                        use_cache=False,
                    )
                    ref_logits_full = ref_outputs.logits[0].to(input_ids.device)   # [L_noisy, V]

                ref_logits_all = ref_logits_full.index_select(0, ref_pos)   # [K, V]

                valid_mask = ~padding_mask
                tau_ref = getattr(self.args, "ref_temperature", 1.0)

                if valid_mask.any():
                    student_log_prob = F.log_softmax(
                        student_logits_all[valid_mask].float() / tau_ref,
                        dim=-1,
                    )
                    teacher_prob = F.softmax(
                        ref_logits_all[valid_mask].float() / tau_ref,
                        dim=-1,
                    )
                    loss_ref = F.kl_div(
                        student_log_prob,
                        teacher_prob,
                        reduction="batchmean",
                    ) * (tau_ref * tau_ref)
                else:
                    loss_ref = torch.zeros((), device=self.args.device)
            else:
                loss_ref = torch.zeros((), device=self.args.device)

        # =========================================================
        # Total loss switch
        #   with ref_model: consistency + ref loss
        #   without ref_model: consistency + ar loss
        # =========================================================
        if self.ref_model is not None:
            ref_weight = getattr(self.args, "ref_weight", 1.0)
            if ref_weight == 0.0:
                print("[warning] ref_weight=0.0, consistency loss only", flush=True)
                total_loss = loss_consistency
            else:
                print(f"[info] ref_weight={ref_weight}, combining consistency and ref losses", flush=True)
                total_loss = loss_consistency + ref_weight * loss_ref
        else:
            print("[info] no ref model, using AR loss as auxiliary to consistency loss", flush=True)
            total_loss = loss_consistency + loss_ar

        if self.args.qlora:
            total_loss.requires_grad = True

        if self.args.local_rank == 0:
            log_dict = {
                "consistency loss": float(loss_consistency.detach().cpu()),
                "ar loss": float(loss_ar.detach().cpu()),
                "ref loss": float(loss_ref.detach().cpu()),
                "total loss": float(total_loss.detach().cpu()),
            }
            if self.ref_model is not None:
                log_dict["ref loss"] = float(loss_ref.detach().cpu())
            wandb.log(log_dict)

        torch.cuda.empty_cache()

        with self.accelerator.accumulate(model):
            self.accelerator.backward(total_loss)

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        return total_loss.detach()