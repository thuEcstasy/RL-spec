export CUDA_VISIBLE_DEVICES=2,3
export WANDB_PROJECT=RL-spec-jf
export WANDB_MODE=online
export WANDB_API_KEY=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
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"

model_path="/mnt/szf_temp/huggingface/JacobiForcing_Coder_7B_v1"
ref_model_path="/mnt/szf_temp/huggingface/Qwen2.5-Coder-7B-Instruct"
#trajectory_file="/checkpoint/lhu/data/CLLM2_openthought/merged/merged_data_v2_8_27_opencodeinstruct.jsonl"
prompt_file="/mnt/szf_temp/datasets/OpenCodeInstruct/data/train-00011-of-00050.jsonl"
output_path="/mnt/szf_temp/JacobiForcing/JacobiForcing/outputs/rl_spec_ref0.5_n32w16"
n_token_seq_size=32
qlora=False

torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=101 \
    --rdzv_endpoint='localhost:5667' \
    --master_port 10000 \
    train/soft_flexattn_train_rl_spec.py \
    --target_model_path ${model_path} \
    --ref_model_path ${ref_model_path} \
    --rollout_model_path ${model_path} \
    --data_path ${prompt_file} \
    --output_dir ${output_path} \
    --max_new_tokens ${n_token_seq_size} \
    --bf16 True \
    --report_to wandb \
    --do_train \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing True \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 8 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --model_max_length 16384 \
    --qlora ${qlora} \
    --ref_weight 0.5 \
