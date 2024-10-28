lm_eval --model hf \
    --model_args pretrained=EleutherAI/gpt-j-6B \
    --tasks winogrande,ai2_arc,hellaswag,truthfulqa_mc2,mmlu \
    --device cuda:0 \
    --batch_size 8 \
    --output_path