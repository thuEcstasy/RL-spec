# online_jacobi_dataset.py
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import torch
from torch.utils.data import IterableDataset

import transformers

from pathlib import Path
import sys
\
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from modeling.cllm2_qwen2_modeling_kv_terminate_on_eos_improved import jacobi_forward_greedy



def _pad_or_truncate_block(x: torch.Tensor, target_len: int, pad_id: int) -> torch.Tensor:
    cur_len = x.shape[1]
    if cur_len == target_len:
        return x
    if cur_len > target_len:
        return x[:, :target_len]
    pad = torch.full((1, target_len - cur_len), pad_id, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=1)

@dataclass
class OnlineJacobiConfig:
    n_token_seq_len: int = 64
    max_new_tokens: int = 512
    max_calls: int = 128
    alt_eos_id: Optional[int] = 151645
    random_init_from_prompt: bool = True

    # 这个开关很关键：
    # 如果你的 trainer 期待 input_ids 只包含“continuation/trajectory tokens”，就设 False
    # 如果 trainer 期待 input_ids 是“prompt + continuation 的完整序列”，就设 True
    include_prompt_in_input_ids: bool = False

    # 为了避免 dataloader worker 进程里碰 GPU，online sampling 必须单进程
    require_num_workers_zero: bool = True


class PromptOnlyDataset(torch.utils.data.Dataset):
    """
    只提供 prompt，不提供离线 trajectory。
    支持三种输入格式：
      1) 样本里已经有 prompt_ids
      2) 样本里有 text/prompt 字段，需要 tokenizer 编码
      3) 样本里有 messages，需要 apply_chat_template
    """
    def __init__(
        self,
        raw_data,
        tokenizer: transformers.PreTrainedTokenizer,
        model_max_length: int,
    ):
        self.raw_data = raw_data
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length

    def __len__(self):
        return len(self.raw_data)

    def _encode_from_text(self, text: str) -> torch.Tensor:
        out = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.model_max_length,
            add_special_tokens=True,
        )
        return out["input_ids"][0]

    def __getitem__(self, idx) -> Dict[str, Any]:
        row = self.raw_data[idx]

        # case 1: already tokenized
        if "prompt_ids" in row:
            prompt_ids = torch.tensor(row["prompt_ids"], dtype=torch.long)
            prompt_len = int(row.get("prompt_ids_len", len(prompt_ids)))
            return {
                "prompt_ids": prompt_ids,
                "prompt_ids_len": prompt_len,
                "meta": row,
            }

        # case 2: chat messages
        if "messages" in row:
            text = self.tokenizer.apply_chat_template(
                row["messages"],
                tokenize=False,
                add_generation_prompt=True,
            )
            
            prompt_ids = self._encode_from_text(text)
            return {
                "prompt_ids": prompt_ids,
                "prompt_ids_len": int(prompt_ids.numel()),
                "meta": row,
            }

        # case 3: plain text prompt
        if "prompt" in row:
            prompt_text = row["prompt"]
        elif "text" in row:
            prompt_text = row["text"]
        elif "input" in row:
            prompt_text = row["input"]
        else:
            raise KeyError(
                "PromptOnlyDataset expects one of: prompt_ids / messages / prompt / text / input"
            )
        prompt_text = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": prompt_text}
            ],
            tokenize=False,
            add_generation_prompt=True
        )
        
        prompt_ids = self._encode_from_text(prompt_text)
        return {
            "prompt_ids": prompt_ids,
            "prompt_ids_len": int(prompt_ids.numel()),
            "meta": row,
        }


class OnlineJacobiTrajectoryDataset(IterableDataset):
    """
    核心点：
      - 数据集只保存 prompts
      - 真正取样时，调用当前 model 做 jacobi rollout
      - 生成一个 trainer 可直接消费的样本字典

    注意：
      1) 必须 num_workers=0
      2) 需要在 accelerator.prepare(...) 之后调用 set_model(...)
      3) 每个 rank 只处理自己的数据分片
    """
    def __init__(
        self,
        prompt_dataset: PromptOnlyDataset,
        tokenizer: transformers.PreTrainedTokenizer,
        cfg: OnlineJacobiConfig,
    ):
        super().__init__()
        self.prompt_dataset = prompt_dataset
        self.tokenizer = tokenizer
        self.cfg = cfg

        self.rollout_model = None
        self.rank = 0
        self.world_size = 1
        self.global_idx = 0
        self.noise_schedule = [13, 12, 11, 10, 9, 8, 7, 6, 5, 4] # TODO: maybe tune this

        # 给不同 epoch 做打乱用
        self.epoch = 0

    def __len__(self):
        # 一个 epoch 里，每个 rank 只处理自己那份 shard
        n = len(self.prompt_dataset)
        return (n + self.world_size - 1) // self.world_size

    def set_rollout_model(self, rollout_model):
        self.rollout_model = rollout_model

        if not hasattr(self.rollout_model.__class__, "jacobi_forward_greedy"):
            self.rollout_model.__class__.jacobi_forward_greedy = jacobi_forward_greedy

    def set_distributed(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def _get_generation_model(self):
        if self.rollout_model is None:
            raise RuntimeError("Call dataset.set_rollout_model(...) first")
        return self.rollout_model

    def _sample_random_tokens_from_history(
        self,
        history_ids: torch.Tensor,
        length: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        history_ids: [1, T]
        return: [1, length]
        """
        flat = history_ids[0].tolist()
        if len(flat) == 0:
            eos = self.tokenizer.eos_token_id or 0
            return torch.full((1, length), eos, dtype=torch.long, device=device)

        out = []
        for _ in range(length):
            tok = random.choice(flat)
            out.append(tok)
        return torch.tensor(out, dtype=torch.long, device=device).unsqueeze(0)

    def _hit_eos(self, generated_part: torch.Tensor) -> bool:
        eos_id = self.tokenizer.eos_token_id
        hit = False
        if eos_id is not None:
            hit = (generated_part == eos_id).any().item()
        if (not hit) and (self.cfg.alt_eos_id is not None):
            hit = (generated_part == self.cfg.alt_eos_id).any().item()
        return bool(hit)

    @torch.no_grad()
    def _rollout_one(self, prompt_ids_1d: torch.Tensor) -> Dict[str, Any]:
        print("start rollout...!!!!!", flush=True)
        model = self._get_generation_model()
        model.eval()

        device = next(model.parameters()).device
        prompt_ids = prompt_ids_1d.to(device).unsqueeze(0)
        attention_mask = torch.ones_like(prompt_ids, device=device)

        n_token_seq_len = self.cfg.n_token_seq_len
        max_new_tokens = self.cfg.max_new_tokens
        max_calls = self.cfg.max_calls

        prompt_len = prompt_ids.shape[1]
        generated_ids = prompt_ids.clone()

        prefill_phase = True
        prefill_drafted_n_gram = None
        past_key_values = None
        first_correct_token = None

        total_new_tokens = 0
        calls = 0
        traj_position_indices: List[int] = []
        draft_blocks: List[torch.Tensor] = []
        accept_blocks: List[torch.Tensor] = []
        itr_list: List[int] = []
        
        while True:
            generated_part = generated_ids[:, prompt_len:]
            if self._hit_eos(generated_part[0]):
                break
            if total_new_tokens >= max_new_tokens:
                break
            if calls >= max_calls:
                break

            if prefill_phase:
                # 和你推理脚本一致：prefill 时随机初始化一个 n_token_seq_len draft
                prefill_draft_token_ids = self._sample_random_tokens_from_history(
                    generated_ids,
                    n_token_seq_len,
                    device,
                )
                prefill_input_ids = torch.cat((prompt_ids, prefill_draft_token_ids), dim=-1)

                past_key_values, first_correct_token, prefill_drafted_n_gram, _, _ = model.jacobi_forward_greedy(
                    input_ids=prefill_input_ids,
                    attention_mask=attention_mask,
                    past_key_values=None,
                    use_cache=True,
                    prefill_phase=True,
                    n_token_seq_len=n_token_seq_len,
                    tokenizer=self.tokenizer,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

                prefill_phase = False
                calls += 1
                continue

            # generation phase
            if calls == 1:
                # 第一轮 generation，沿用 prefill 返回的 drafted n-gram
                step_input_ids = prefill_drafted_n_gram
            else:
                random_tail = self._sample_random_tokens_from_history(
                    generated_ids,
                    n_token_seq_len - 1,
                    device,
                )
                step_input_ids = torch.cat((first_correct_token.view(1, -1), random_tail), dim=-1)

            accepted_history_ids = generated_ids[:, prompt_len:]
            past_key_values, first_correct_token, accepted_n_gram, itr, noisy_block_record = model.jacobi_forward_greedy(
                input_ids=step_input_ids,
                attention_mask=None,
                past_key_values=past_key_values,
                use_cache=True,
                prefill_phase=False,
                n_token_seq_len=n_token_seq_len,
                tokenizer=self.tokenizer,
                accepted_history_ids=accepted_history_ids,
                eos_token_id=self.tokenizer.eos_token_id,
                capture_noisy_block=True,
                capture_len=n_token_seq_len * self.noise_schedule[self.global_idx % len(self.noise_schedule)] // 16, # TODO: maybe tune this
            )
            self.global_idx += 1
            if accepted_n_gram is None or accepted_n_gram.numel() == 0:
                break

            generated_ids = torch.cat((generated_ids, accepted_n_gram), dim=-1)
            itr_list.append(itr)
            if noisy_block_record is not None:
                draft_blocks.append(noisy_block_record["noisy_block"].clone())         # full noisy block
                accept_blocks.append(accepted_n_gram.clone())   
            total_new_tokens = generated_ids.shape[1] - prompt_len
            traj_position_indices.append(total_new_tokens)
            calls += 1
        tokens_per_iter = total_new_tokens / sum(itr_list) if itr_list and total_new_tokens > 0 else None
        print("finished rollout! total_new_tokens =", total_new_tokens, "tokens/iter = ", tokens_per_iter, flush=True)
        return {
            "prompt_ids": prompt_ids,
            "prompt_ids_len": prompt_len,
            "final_generated_ids": generated_ids,
            "draft_blocks": draft_blocks,
            "accept_blocks": accept_blocks,
            "tokens_per_iter": tokens_per_iter
        }


    def _build_training_sample(self, prompt_sample: Dict[str, Any]) -> Dict[str, Any]:
        rollout = self._rollout_one(prompt_sample)

        prompt_ids_2d = rollout["prompt_ids"]          # [1, P]
        prompt_len = rollout["prompt_ids_len"]
        draft_blocks = rollout["draft_blocks"]         # list of [1, N]
        accept_blocks = rollout["accept_blocks"]       # list of [1, capture_len]

        N = self.cfg.n_token_seq_len
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0

        seq_parts = [prompt_ids_2d.squeeze(0)]

        T = min(len(draft_blocks), len(accept_blocks))
        print("T:", T, flush=True)
        for j in range(T):
            k_j = _pad_or_truncate_block(draft_blocks[j], N, pad_id).squeeze(0)
            l_j = _pad_or_truncate_block(accept_blocks[j], N, pad_id).squeeze(0)
            seq_parts.append(k_j)
            seq_parts.append(l_j)

        input_ids = torch.cat(seq_parts, dim=0).cpu()

        # 你的 trainer 目前只用这个字段的长度来得到 T
        traj_position_indices = torch.arange(T, dtype=torch.long).unsqueeze(0)
        if T == 0:
            return None
        else:
            return {
                "prompt_ids": prompt_ids_2d.squeeze(0).cpu(),
                "prompt_ids_len": int(prompt_len),
                "input_ids": input_ids,
                "traj_position_indices": traj_position_indices,
                "tokens_per_iter": rollout["tokens_per_iter"],
            }

    def __iter__(self) -> Iterable[Dict[str, Any]]:

        # rank-wise shard
        indices = list(range(len(self.prompt_dataset)))
        rng = random.Random(self.epoch + 12345)
        rng.shuffle(indices)

        rank_indices = indices[self.rank::self.world_size]
        for idx in rank_indices:
            prompt_sample = self.prompt_dataset[idx]
            # sample = self._build_training_sample(prompt_sample)
            # if sample is None:
            #     print(f"[dataset] skip sample idx={idx} because T=0", flush=True)
            #     continue
            prompt_sample["sample_idx"] = torch.tensor(idx, dtype=torch.long)
            yield prompt_sample

