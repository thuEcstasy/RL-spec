from dataclasses import dataclass
from typing import List, Dict, Any
import torch

from dataclasses import dataclass
from typing import List, Dict, Any
import torch
import transformers


@dataclass
class PromptOnlyCollator:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            raise ValueError("tokenizer.pad_token_id and eos_token_id are both None")

        prompt_ids_list = [f["prompt_ids"] for f in features]
        prompt_lens = torch.tensor([f["prompt_ids_len"] for f in features], dtype=torch.long)
        metas = [f["meta"] for f in features]

        max_len = max(x.size(0) for x in prompt_ids_list)

        padded_prompt_ids = []
        attention_masks = []

        for x in prompt_ids_list:
            pad_len = max_len - x.size(0)

            if pad_len > 0:
                pad = torch.full((pad_len,), pad_token_id, dtype=x.dtype)
                padded_x = torch.cat([x, pad], dim=0)
                attn = torch.cat(
                    [torch.ones(x.size(0), dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)],
                    dim=0,
                )
            else:
                padded_x = x
                attn = torch.ones(x.size(0), dtype=torch.long)

            padded_prompt_ids.append(padded_x)
            attention_masks.append(attn)

        return {
            "prompt_ids": torch.stack(padded_prompt_ids, dim=0),      # [B, Lmax]
            "prompt_ids_len": prompt_lens,                            # [B]
            "prompt_attention_mask": torch.stack(attention_masks, 0), # [B, Lmax]
        }

@dataclass
class CustomCollator:
    tokenizer: any
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 1. 先把自定义字段拿出去
        prompt_ids_list = [f.pop("prompt_ids") for f in features] if "prompt_ids" in features[0] else None
        traj_pos_list = [f.pop("traj_position_indices") for f in features] if "traj_position_indices" in features[0] else None
        labels_list = [f.pop("labels") for f in features] if "labels" in features[0] else None

        # 2. 这里只保留 tokenizer 能处理的标准字段
        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )

        # 3. labels 单独 pad
        if labels_list is not None:
            labels_list = [torch.tensor(x, dtype=torch.long) for x in labels_list]
            batch["labels"] = torch.nn.utils.rnn.pad_sequence(
                labels_list,
                batch_first=True,
                padding_value=self.label_pad_token_id,
            )

        # 4. prompt_ids 单独 pad
        if prompt_ids_list is not None:
            prompt_ids_list = [torch.tensor(x, dtype=torch.long) for x in prompt_ids_list]
            batch["prompt_ids"] = torch.nn.utils.rnn.pad_sequence(
                prompt_ids_list,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            )

        # 5. traj_position_indices 单独 pad
        traj_tensors = []
        for x in traj_pos_list:
            t = torch.tensor(x, dtype=torch.long)
            if t.dim() > 1:
                t = t.reshape(-1)   # 或者 t.squeeze(0)，但 reshape(-1) 更稳
            traj_tensors.append(t)

        batch["traj_position_indices"] = torch.nn.utils.rnn.pad_sequence(
            traj_tensors,
            batch_first=True,
            padding_value=-1,  # 用 -1 来区分 padding 的位置
        )
        return batch