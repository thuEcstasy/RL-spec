#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from torch.utils.data import Dataset, DataLoader
import torch
import transformers
from transformers.trainer_pt_utils import LabelSmoother
from dataclasses import dataclass, field
from typing import Dict, Optional, List
from datasets import load_dataset
import logging
import math
import os
import sys
import pathlib

from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import DataLoaderConfiguration
from transformers.integrations import HfDeepSpeedConfig
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

from pathlib import Path
path_root = Path(__file__).parents[0]
sys.path.append(str(path_root))

from soft_flexattn_cllm_trainer_multiblock import CllmTrainer
from online_jacobi_dataset import (
    PromptOnlyDataset,
    OnlineJacobiTrajectoryDataset,
    OnlineJacobiConfig,
)
from custom_collator import CustomCollator, PromptOnlyCollator

logger = logging.getLogger(__name__)
IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    target_model_path: Optional[str] = field(
        default="models/vicuna-7b-v1.5",
        metadata={"help": "Path or HF id for the base model"},
    )
    ref_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path or HF id for the ref model; if not specified, no ref loss will be used"},
    )
    qlora: bool = field(default=False, metadata={"help": "Enable QLoRA"})


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data JSONL"})
    lazy_preprocess: bool = True
    online_jacobi_sampling: bool = field(
        default=True,
        metadata={"help": "Whether to do on-the-fly Jacobi rollout instead of loading offline trajectories"},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=2048, metadata={"help": "Max seq length for tokenizer/model"})
    max_new_tokens: int = field(default=64, metadata={"help": "N_BLOCK in the Jacobi trajectory"})
    report_to: str = field(default="wandb")
    use_gt_labels: bool = field(default=False)
    remove_unused_columns: bool = field(default=False)
    gradient_checkpointing: bool = field(default=True)
    bf16: bool = field(default=True)
    fp16: bool = field(default=False)

    max_pairs_per_step: int = field(
        default=10,
        metadata={"help": "How many (k_j, last_j) pairs to use per training step (<= 0 means use all)"},
    )

    # online jacobi sampling args
    online_jacobi_max_new_tokens: int = field(default=512)
    online_jacobi_max_calls: int = field(default=128)
    online_jacobi_alt_eos_id: int = field(default=151645)
    online_jacobi_include_prompt_in_input_ids: bool = field(default=False)

    # rollout / sampler model args
    rollout_attn_implementation: str = field(default="flash_attention_2")
    rollout_model_path: Optional[str] = field(default=None)
    rollout_sync_every: int = field(default=1)
    rollout_device: Optional[str] = field(default=None)
    rollout_bf16: bool = field(default=True)
    ref_weight: float = field(default=10.0, metadata={"help": "Weight for the ref model in the loss; 0 means no ref model"})

def rank0_print(local_rank, *args):
    if local_rank in (0, -1):
        print(*args, flush=True)


def safe_save_model_for_hf_trainer(model, tokenizer, output_dir, accelerator=None):
    os.makedirs(output_dir, exist_ok=True)
    to_save = model
    if accelerator is not None:
        to_save = accelerator.unwrap_model(model)
    if hasattr(to_save, "module"):
        to_save = to_save.module
    to_save.save_pretrained(output_dir, safe_serialization=False)
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)


def preprocess_training_sequence(
    prompt_ids: List[int],
    prompt_ids_len: int,
    complete_training_sequence_ids: List[int],
    tokenizer: transformers.PreTrainedTokenizer,
    model: str,
    traj_position_indices: List[int],
) -> Dict:
    prompt_ids = torch.tensor(prompt_ids, dtype=torch.long)
    complete_training_sequence_ids = torch.tensor(complete_training_sequence_ids, dtype=torch.long)
    return dict(
        prompt_ids=prompt_ids,
        prompt_ids_len=prompt_ids_len,
        input_ids=complete_training_sequence_ids,
        traj_position_indices=torch.tensor(traj_position_indices, dtype=torch.long).unsqueeze(0),
    )


def make_online_jacobi_data_module(tokenizer, prompt_data, training_args) -> Dict:
    prompt_dataset = PromptOnlyDataset(
        raw_data=prompt_data,
        tokenizer=tokenizer,
        model_max_length=training_args.model_max_length,
    )

    online_cfg = OnlineJacobiConfig(
        n_token_seq_len=training_args.max_new_tokens,
        max_new_tokens=training_args.online_jacobi_max_new_tokens,
        max_calls=training_args.online_jacobi_max_calls,
        alt_eos_id=training_args.online_jacobi_alt_eos_id,
        include_prompt_in_input_ids=training_args.online_jacobi_include_prompt_in_input_ids,
    )

    train_dataset = OnlineJacobiTrajectoryDataset(
        prompt_dataset=prompt_dataset,
        tokenizer=tokenizer,
        cfg=online_cfg,
    )
    return dict(train_dataset=train_dataset, eval_dataset=None)


class JacobianDataset(Dataset):
    def __init__(self, raw_data, tokenizer, model: str, do_eval: bool = False, local_rank: int = -1):
        super().__init__()
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}
        self.do_eval = do_eval
        self.model = model
        rank0_print(local_rank, "Formatting inputs (lazy)...")

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]
        sample = self.raw_data[i]
        ret = preprocess_training_sequence(
            sample["prompt_ids"],
            sample["prompt_ids_len"],
            sample["complete_training_sequence_ids"],
            self.tokenizer,
            self.model,
            traj_position_indices=sample["traj_position_indices"],
        )
        self.cached_data_dict[i] = ret
        return ret


def make_jacobian_data_module(tokenizer, trajectory_data, data_args, model: str, local_rank: int) -> Dict:
    assert data_args.lazy_preprocess, "only support lazy preprocess"
    train_dataset = JacobianDataset(trajectory_data, tokenizer=tokenizer, model=model, local_rank=local_rank)
    return dict(train_dataset=train_dataset, eval_dataset=None)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    training_args.local_rank = local_rank
    training_args.qlora = model_args.qlora
    training_args.deepspeed = training_args.deepspeed or "scripts/train/ds_config_cpu_offloading.json"

    # -------------------------
    # config / tokenizer
    # -------------------------
    config = transformers.AutoConfig.from_pretrained(
        model_args.target_model_path,
        cache_dir=training_args.cache_dir,
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        raise ValueError(
            f"Use rope scaling is not supported: "
            f"model_max_length ({training_args.model_max_length}) > orig_ctx_len ({orig_ctx_len})"
        )
    config.use_cache = False

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.target_model_path,
        padding_side="right",
        use_fast=False,
    )

    # -------------------------
    # load rollout and ref model FIRST
    # no deepspeed / no accelerator / inference-only
    # -------------------------
    
    ref_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.ref_model_path if model_args.ref_model_path is not None else model_args.target_model_path,
        config=config,
        cache_dir=training_args.cache_dir,
        attn_implementation=training_args.rollout_attn_implementation,
        torch_dtype=torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else None),
        low_cpu_mem_usage=True,
    )
    ref_device = training_args.rollout_device
    if ref_device is None:
        if torch.cuda.is_available():
            if local_rank >= 0:
                ref_device = f"cuda:{local_rank}"
            else:
                ref_device = "cuda:0"
        else:
            ref_device = "cpu"

    ref_model.to(ref_device)
    ref_model.eval()
    ref_model.config.use_cache = True
    
    rollout_model = None
    if data_args.online_jacobi_sampling:
        rollout_model_path = training_args.rollout_model_path or model_args.target_model_path
        rollout_dtype = torch.bfloat16 if training_args.rollout_bf16 else (
            torch.float16 if training_args.fp16 else None
        )

        rollout_model = transformers.AutoModelForCausalLM.from_pretrained(
            rollout_model_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=training_args.rollout_attn_implementation,
            torch_dtype=rollout_dtype,
            low_cpu_mem_usage=False,  # important for debugging / avoid weird empty shards
        )

        rollout_device = training_args.rollout_device
        if rollout_device is None:
            if torch.cuda.is_available():
                if local_rank >= 0:
                    rollout_device = f"cuda:{local_rank}"
                else:
                    rollout_device = "cuda:0"
            else:
                rollout_device = "cpu"

        rollout_model.to(rollout_device)
        rollout_model.eval()
        rollout_model.config.use_cache = True

    # -------------------------
    # now create accelerator / deepspeed for TRAIN model only
    # -------------------------
    ds_plugin = DeepSpeedPlugin(hf_ds_config=training_args.deepspeed)
    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps or 1,
        mixed_precision="bf16" if training_args.bf16 else ("fp16" if training_args.fp16 else "no"),
        log_with=training_args.report_to if training_args.report_to else None,
        deepspeed_plugin=ds_plugin,
        dataloader_config=DataLoaderConfiguration(
            dispatch_batches=False,
        ),
    )

    # ZeRO-3/offload compatibility for training model only
    dschf = HfDeepSpeedConfig(training_args.deepspeed)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO if accelerator.is_local_main_process else logging.ERROR,
    )
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        f"distributed training: {bool(training_args.local_rank != -1)}, bf16: {training_args.bf16}, fp16: {training_args.fp16}"
    )
    logger.info(f"Training/eval parameters {training_args}")

    # -------------------------
    # training model
    # -------------------------
    attn_impl = "flex_attention"
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.target_model_path,
        config=config,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_impl,
        torch_dtype=torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else None),
        low_cpu_mem_usage=True,
    )


    if getattr(training_args, "gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    if model_args.qlora:
        model = prepare_model_for_kbit_training(model)
        lora_cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=32, lora_alpha=16, lora_dropout=0.05)
        model = get_peft_model(model, lora_cfg)

    # -------------------------
    # data
    # -------------------------
    raw_dataset = load_dataset(
        "json",
        data_files={"train": data_args.data_path},
        split="train",
        cache_dir=training_args.cache_dir,
    )

    if data_args.online_jacobi_sampling:
        data_module = make_online_jacobi_data_module(
            tokenizer=tokenizer,
            prompt_data=raw_dataset,
            training_args=training_args,
        )

        if training_args.dataloader_num_workers != 0:
            logger.warning("online_jacobi_sampling=True requires dataloader_num_workers=0; overriding to 0")
            training_args.dataloader_num_workers = 0

        dataset = data_module["train_dataset"]
        dataset.set_rollout_model(rollout_model)
        dataset.set_distributed(
            rank=accelerator.process_index,
            world_size=accelerator.num_processes,
        )
        train_dataloader = DataLoader(
            dataset,
            batch_size=training_args.per_device_train_batch_size or 1,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
            collate_fn=PromptOnlyCollator(tokenizer),
        )
    else:
        data_module = make_jacobian_data_module(
            tokenizer=tokenizer,
            trajectory_data=raw_dataset,
            data_args=data_args,
            model=model_args.target_model_path,
            local_rank=training_args.local_rank,
        )

        train_dataloader = DataLoader(
            data_module["train_dataset"],
            batch_size=training_args.per_device_train_batch_size or 1,
            shuffle=False,
            num_workers=training_args.dataloader_num_workers,
            pin_memory=True,
            drop_last=False,
        )

    # -------------------------
    # optimizer / scheduler
    # -------------------------
    params = (p for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(
        params,
        lr=training_args.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    if data_args.online_jacobi_sampling:
        train_dataloader_len = len(data_module["train_dataset"])
    else:
        train_dataloader_len = len(train_dataloader)
    num_update_steps_per_epoch = math.ceil(
        train_dataloader_len / (training_args.gradient_accumulation_steps or 1)
    )
    if training_args.max_steps > 0:
        max_train_steps = training_args.max_steps
    else:
        max_train_steps = int(training_args.num_train_epochs * num_update_steps_per_epoch)

    lr_scheduler = transformers.get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_args.get_warmup_steps(max_train_steps),
        num_training_steps=max_train_steps,
    )

    if data_args.online_jacobi_sampling:
        # IterableDataset already shards by rank in __iter__; skip accelerator sharding
        model, optimizer, lr_scheduler = accelerator.prepare(
            model, optimizer, lr_scheduler
        )
    else:
        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, lr_scheduler
        )

    trainer = CllmTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=data_module["train_dataset"],
        accelerator=accelerator,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_dataloader=train_dataloader,
        ref_model=ref_model,
    )

    if data_args.online_jacobi_sampling:
        trainer.rollout_dataset = data_module["train_dataset"]
    else:
        trainer.rollout_dataset = None

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    if accelerator.is_main_process:
        safe_save_model_for_hf_trainer(
            model=model,
            tokenizer=tokenizer,
            output_dir=training_args.output_dir,
            accelerator=accelerator,
        )


if __name__ == "__main__":
    train()