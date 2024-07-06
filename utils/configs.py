import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task")

    parser.add_argument('--data_dir', default=None, type=str, help='')
    parser.add_argument('--task_name', default=None, type=str, help='')
    parser.add_argument('--data_file', default=None, type=str, help='')
    parser.add_argument('--max_length', type=int, default=784, help='')
    parser.add_argument('--max_src_len', type=int, default=768, help='')

    parser.add_argument('--output_dir', default=None, type=str, help='')

    parser.add_argument('--model_name_or_path', default=None, type=str, help='')
    parser.add_argument('--tokenizer_name', default="THUDM/chatglm2-6b", type=str, help='')

    parser.add_argument('--seed', default=42, type=int, help='')
    parser.add_argument('--num_train_epochs', default=5, type=int, help='')
    parser.add_argument('--per_train_batch_size', default=2, type=int, help='')
    parser.add_argument('--per_valid_batch_size', default=2, type=int, help='')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help='')
    parser.add_argument('--log_steps', type=int, default=100, help='')
    parser.add_argument('--local_rank', type=int, default=0, help='')

    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--over_cache', type=bool, default=False, help='')
    parser.add_argument('--lora_r', type=int, default=8, help='')

    args = parser.parse_args()

    return args


def build_configs():
    args = parse_args()

    # Initialize the path.
    args.output_dir = os.path.join(args.output_dir, args.task_name)
    os.makedirs(args.output_dir, exist_ok=True)

    if not args.data_file:
        args.data_file = os.path.join(args.data_dir, f"{args.task_name}.pkl")
    if not os.path.isfile(args.data_file):
        raise ValueError(f"{args.task_name} couldn't be found in {args.data_file}")

    return args


def ds_configs(args):
    conf = {
        "train_micro_batch_size_per_gpu": args.per_train_batch_size,
        "eval_micro_batch_size_per_gpu": args.per_valid_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": "auto",
                "betas": "auto",
                "eps": "auto",
                "weight_decay": "auto"
            }
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "last_batch_iteration": -1,
                "total_num_steps": "auto",
                "warmup_min_lr": "auto",
                "warmup_max_lr": "auto",
                "warmup_num_steps": "auto",
                "warmup_type": "cosine"
            }
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        "bf16": {
            "enabled": False
        },
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 100
        },
        "zero_optimization": {
            "stage": 1,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True
        },
        "steps_per_print": args.log_steps
    }

    return conf
