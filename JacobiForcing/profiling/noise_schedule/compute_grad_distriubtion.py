from pathlib import Path
import sys
from datasets import load_dataset
import transformers
import torch
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from train.soft_flexattn_train_rl_spec import make_online_jacobi_data_module 
from train.soft_flexattn_train_rl_spec import ModelArguments, DataArguments, TrainingArguments
data_path = "/mnt/szf_temp/datasets/OpenCodeInstruct/data/first_10000.jsonl"
rollout_model_path = "/mnt/szf_temp/huggingface/JacobiForcing_Coder_7B_v1"
input_path = "/mnt/szf_temp/datasets/OpenCodeInstruct/data/first_10000.jsonl"

parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()

raw_dataset = load_dataset(
    "json",
    data_files={"train": data_path},
    split="train",
)

rollout_model = transformers.AutoModelForCausalLM.from_pretrained(
    rollout_model_path,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,  # important for debugging / avoid weird empty shards
)
rollout_model.to("cuda")
rollout_model.eval()

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
    output = dataset._build_training_sample(sample["prompt_ids"])