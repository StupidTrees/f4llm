import os
import json
import time
import argparse
import pandas as pd
from glob import glob
from loguru import logger
from tabulate import tabulate
from multiprocessing import Pool
from utils.general import read_json, setup_seed, run_process


task_group = {
    "alpaca": "winogrande,ai2_arc,hellaswag,truthfulqa_mc2,mmlu",
    "alpacare": "mmlu_clinical_knowledge,mmlu_anatomy,mmlu_college_medicine,"
                "mmlu_college_biology,mmlu_nutrition,mmlu_virology,"
                "mmlu_medical_genetics,mmlu_professional_medicine",

}

task_mertic_mapping = {
    "winogrande": "acc,none",
    'truthfulqa_mc2': 'acc,none',
    'mmlu': 'acc,none',
    "hellaswag": "acc_norm,none",
    "ai2_arc": "acc_norm,none"
}


def get_task(task_name):
    if task_name in ["alpaca", "alpacare"]:
        tasks = task_group[task_name].split(",")
    else:
        # single task eval
        tasks = task_name.split(",")
    return tasks


def get_model_path(model_type, run_dirs):
    if model_type == "llama2-base":
        model_path = f"{run_dirs}/pretrain/nlp/Llama-2-7b-hf/"
    elif model_type == "llama2-chat":
        model_path = f"{run_dirs}/pretrain/nlp/Llama-2-7b-chat-hf/"
    elif model_type == "tinyllama":
        model_path = f"{run_dirs}/pretrain/nlp/TinyLlama-1.1B-intermediate-step-1195k-token-2.5T/"
    elif model_type == "qwen":
        model_path = f"{run_dirs}/pretrain/nlp/Qwen-1_8B/"
    else:
        raise ValueError
    return model_path


def merge_metric(result_files, tasks, output_path, task_name):
    mean_metrics = {}
    best_metrics = {}
    for result_file in result_files:
        checkpoint_name = result_file.split("/")[-2]
        mean_metrics[checkpoint_name] = {}
        # checkpoint_path = "/".join(result_file.split("/")[0:-1])
        # logger.info(f">> Test result_file: {checkpoint_name}")

        metrics = read_json(result_file)["results"]
        for task in tasks:
            metric_name = task_mertic_mapping[task]
            metric = round(metrics[task][metric_name]*100, 1)
            mean_metrics[checkpoint_name][task] = metric
            if task not in best_metrics or best_metrics[task] < metric:
                best_metrics[task] = metric

    metrics_df = pd.DataFrame.from_dict(mean_metrics, orient='index')
    metrics_df["mean"] = metrics_df.mean(axis=1).round(1)
    sorted_metrics_df = metrics_df.sort_values(by='mean', ascending=False)
    sorted_metrics_df.reset_index(inplace=True)
    sorted_metrics_df.columns = ["round"] + list(sorted_metrics_df.columns[1:])

    all_metric_path = os.path.join(output_path, f"{task_name}_round_metric.csv")
    sorted_metrics_df.to_csv(all_metric_path, sep="\t", index=False)
    logger.info(f">> Metric outputs saved to {all_metric_path}")

    best_metric_path = os.path.join(output_path, f"{task_name}_best_metric.csv")
    best_metric_df = pd.DataFrame.from_dict({"best": best_metrics}, orient='index')
    best_metric_df["mean"] = best_metric_df.mean(axis=1).round(1)
    # best_metric_df.reset_index(inplace=True)
    best_metric_df.to_csv(best_metric_path, sep="\t", index=False)
    logger.info(f">> Best Metric outputs saved to {best_metric_path}")

    logger.info(f'\n{tabulate(sorted_metrics_df, headers="keys", tablefmt="pretty", showindex=False)}')
    logger.info(f'\n{tabulate(best_metric_df, headers="keys", tablefmt="pretty", showindex=False)}')
    logger.info(f">> Copy Results")
    run_process(f"cat {all_metric_path}")
    print()
    run_process(f"cat {best_metric_path}")


def single_metric(result_file, tasks, output_path, task_name):
    mean_metrics = {}
    checkpoint_name = result_file.split("/")[-2]
    mean_metrics[checkpoint_name] = {}

    metrics = read_json(result_file)["results"]
    for task in tasks:
        metric_name = task_mertic_mapping[task]
        metric = round(metrics[task][metric_name]*100, 1)
        mean_metrics[checkpoint_name][task] = metric

    metrics_df = pd.DataFrame.from_dict(mean_metrics, orient='index')
    metrics_df["mean"] = metrics_df.mean(axis=1).round(1)
    sorted_metrics_df = metrics_df.sort_values(by='mean', ascending=False)
    sorted_metrics_df.reset_index(inplace=True)
    sorted_metrics_df.columns = ["round"] + list(sorted_metrics_df.columns[1:])

    metric_path = os.path.join(output_path, f"{task_name}_metric.csv")
    sorted_metrics_df.to_csv(metric_path, sep="\t", index=False)
    logger.info(f">> Metric outputs saved to {metric_path}")

    logger.info(f'\n{tabulate(sorted_metrics_df, headers="keys", tablefmt="pretty", showindex=False)}')
    logger.info(f">> Copy Results")
    run_process(f"cat {metric_path}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--eval_model_path", type=str, default="-1")
    parser.add_argument("--run_dirs", type=str, default="/userhome")
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="/userhome/output/")
    parser.add_argument("--use_cache", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    setup_seed(42)
    if args.model_path is None:
        model_path = get_model_path(args.model_name, args.run_dirs)
    else:
        model_path = args.model_path

    if args.eval_model_path != "-1":
        pattern = os.path.join(args.eval_model_path, "round-*")
        checkpoint_dirs = sorted(glob(pattern, recursive=True),
                                 key=lambda x: os.path.getctime(x), reverse=False)
        if len(checkpoint_dirs) == 0:  # single checkpoint
            checkpoint_dirs = [args.eval_model_path]
        eval_model_path = args.eval_model_path
    else:
        checkpoint_dirs = ["zst"]
        args.output_dir = os.path.join(args.output_dir, f"zst/{args.model_name}/")
        os.makedirs(args.output_dir, exist_ok=True)
        eval_model_path = args.output_dir

    gpu_list = args.device.split(",")
    n_gpu = len(gpu_list)
    tasks = get_task(args.task_name)
    batch_size = "auto" if args.batch_size == -1 else args.batch_size
    max_batch_size = 64 if "auto" == batch_size else None

    cmds = []
    options = [
        "--model", "hf",
        "--batch_size", str(batch_size),
    ]
    if max_batch_size:
        options.extend(["--max_batch_size", f"{max_batch_size}"])
    if args.use_cache:
        options.extend(["--use_cache", f"{args.use_cache}"])

    cnt = 0
    result_files = []
    for _, checkpoint_dir in enumerate(checkpoint_dirs):
        if checkpoint_dir == "zst":
            output_file_path = os.path.join(args.output_dir, f"{args.task_name}.json")
            model_args = f"pretrained={model_path},load_in_8bit=True,trust_remote_code=True"
        else:
            output_file_path = os.path.join(checkpoint_dir, f"{args.task_name}.json")
            model_args = f"pretrained={model_path},load_in_8bit=True,trust_remote_code=True,peft={checkpoint_dir}"

        result_files.append(output_file_path)
        if os.path.isfile(output_file_path):
            logger.info(f"result {output_file_path} exists, just skip it.")
            continue

        device = gpu_list[cnt % n_gpu]
        eval_tasks = ",".join(tasks) if len(tasks) > 1 else tasks[0]
        options.extend(
            [
                "--model_args", model_args,
                "--tasks", eval_tasks,
                "--output_path", output_file_path,
                "--device", f"cuda:{device}"
            ]
        )

        cmd = "lm_eval " + " ".join(options)
        cmds.append(cmd)
        cnt += 1

    logger.warning(f"run {len(cmds)} llm {args.task_name} tasks from {args.eval_model_path}")
    run_process("sleep 3s")

    if len(cmds) > 0:
        pool = Pool(processes=n_gpu)
        pool.map(run_process, cmds)

    if len(result_files) > 1:
        merge_metric(result_files, tasks, eval_model_path, args.task_name)
    elif len(result_files) == 1:
        single_metric(result_files[0], tasks, eval_model_path, args.task_name)


if __name__ == "__main__":
    main()
