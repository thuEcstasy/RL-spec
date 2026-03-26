cd /home/szf/evalchemy

python -m eval.eval \
    --model hf \
    --tasks HumanEval \
    --model_args "pretrained=/home/szf/huggingface/Qwen2.5-Coder-7B-Instruct" \
    --batch_size 2 \
    --output_path logs