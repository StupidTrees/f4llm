""" general functions for FNLP """

import os
import json
import pickle
import importlib
import math
import multiprocessing

import random
import shutil
from glob import glob

import numpy as np
import psutil
import torch
from peft import get_peft_model_state_dict
from collections import Counter
import torch.nn.functional as F

from utils.constants import petuning_type
from utils.register import registry


def pickle_read(path, read_format="rb"):
    with open(path, read_format) as file:
        obj = pickle.load(file)
    return obj


def pickle_write(obj, path, write_format="wb"):
    with open(path, write_format) as file:
        pickle.dump(obj, file)


def read_json(path_file):
    outputs = []
    with open(path_file, "r") as file:
        if path_file.endswith("jsonl"):
            for line in file:
                outputs.append(json.loads(line))
        else:
            outputs = json.load(file)
    return outputs


def write_json(obj, path_file):
    with open(path_file, "w") as file:
        if path_file.endswith("jsonl"):
            for line in obj:
                json.dump(line, file)
                file.write('\n')
        else:
            json.dump(obj, file)


def file_write(line, path, mode):
    with open(path, mode) as file:
        file.write(line + "\n")


def make_sure_dirs(path, role="server"):
    """Create dir if not exists

    Args:
        path (str): path
        role (str): sign
    """
    if role == "client":
        return
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


def rm_dirs(path: str):
    """
    remove file existing check.
    Args:
        path (str): path
    """
    if os.path.exists(path):
        shutil.rmtree(path)


def rm_file(file_path: str):
    if os.path.isfile(file_path):
        os.unlink(file_path)


def get_cpus():
    """return total num of cpus in current machine."""
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
            cfs_quota_us = int(f.readline())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
            cfs_period_us = int(f.readline())
        if cfs_quota_us > 0 and cfs_period_us > 0:
            return int(math.ceil(cfs_quota_us / cfs_period_us))
    except Exception:
        pass
    return multiprocessing.cpu_count()


def get_memory_usage():
    """
        return total memory been used.
        memory use in GB
    """
    pid = os.getpid()
    py = psutil.Process(pid)
    memory_use = py.memory_info()[0] / 2.0 ** 30
    return memory_use


def LoadBalanceSampling(target, split_size):
    chunk_size = int(len(target) // split_size)
    result = [target[x:x + chunk_size] for x in range(0, len(target), chunk_size)]

    if len(result) == split_size + 1:
        for i, j in enumerate(result[-1]):
            idx = i % split_size
            result[idx].append(j)
        return result[0:-1]
    elif len(result) == split_size:
        return result
    else:
        raise


def has_repeated_ngrams(text, n=4, threshold=10):
    # 将文本分割成n-gram
    words = text.split()
    ngrams = [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
    # 使用Counter统计n-gram出现的次数
    ngram_counts = Counter(ngrams)
    # print(ngram_counts)
    # 检查是否有n-gram的次数超过阈值
    for count in ngram_counts.values():
        if count > threshold:
            return True
    # 如果没有超过阈值的n-gram，则返回False
    return False


def cosine_learning_rate(current_round, total_rounds, initial_lr=0.001, min_lr=0):
    """
    Compute the learning rate based on a cosine schedule.

    :param current_round: The current training round (0-indexed).
    :param total_rounds: The total number of training rounds.
    :param initial_lr: The initial learning rate.
    :param min_lr: The minimum learning rate.
    :return: The computed learning rate for the current round.
    """
    # Compute the cosine learning rate
    cosine_lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * current_round / total_rounds))
    return cosine_lr


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': round(total_num / 1e6, 4), 'Trainable': round(trainable_num / 1e6, 4)}


def is_petuning(tuning_type):
    for name in petuning_type:
        if name in tuning_type:
            return True
    return False


def get_peft_parameters(model, tuning_type):
    if tuning_type == "adapter":
        peft_model_state_dict = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                peft_model_state_dict[name] = param
    else:
        peft_model_state_dict = get_peft_model_state_dict(model)

    return peft_model_state_dict


def setup_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False  # maybe slower training


def is_best(p_metric, c_metric, low_is_better):
    """
    c_metric: current metric
    p_metric: previous metric
    """
    if not low_is_better and p_metric <= c_metric:
        return True
    elif low_is_better and p_metric >= c_metric:
        return True
    return False


def run_process(proc):
    os.system(proc)


def end_log(fun):
    def wapper(handler_or_trainer, training_config, logger):
        if training_config.local_rank <= 0:
            logger.info(f"see training logs --> {training_config.metric_log_file}")
            logger.info(f"see training results --> {training_config.metric_file}")
            return fun(handler_or_trainer, training_config, logger)

    return wapper


@end_log
def metric_save(trainer, training_config, logger=None):
    pickle_write(trainer.metric_log, training_config.metric_log_file)
    # trainer.metric_line += f"valid_{trainer.metric_name}={trainer.global_valid_best_metric:.3f}_"
    # trainer.metric_line += f"test_{trainer.global_test_best_metric}"
    # file_write(trainer.metric_line, training_config.metric_file, "a+")


def setup_imports():
    from utils.register import registry
    # First, check if imports are already setup
    has_already_setup = registry.get("imports_setup", no_warning=True)
    if has_already_setup:
        return
    # Automatically load all of the modules, so that they register with registry
    root_folder = os.path.dirname(os.path.abspath(__file__))
    project_name = root_folder.split(os.sep)[-2]
    root_folder = os.path.join(root_folder, "..")  # check here
    files = []
    for package_name in ["trainers", "contribs", "models", "datas", "utils", "configs", "evals", "metrics"]:
        folder = os.path.join(root_folder, package_name)
        pattern = os.path.join(folder, "**", "*.py")
        files.extend(glob(pattern, recursive=True))

    for f in files:
        f = os.path.realpath(f)
        if f.endswith(".py") and not f.endswith("__init__.py"):
            splits = f.split(os.sep)
            import_prefix_index = 0
            for idx, split in enumerate(splits):
                if split == project_name:
                    import_prefix_index = idx + 1
            file_name = splits[-1]
            module_name = file_name[: file_name.find(".py")]
            module = ".".join(
                splits[import_prefix_index:-1] + [module_name]
            )
            importlib.import_module(module)

    registry.register("root_folder", root_folder)
    registry.register("imports_setup", True)


def compute_energy(model, tokenizer, new_data, temperature=1.0, top_k=10):
    energy_dps = []
    for dp in new_data:
        sequence = "## Instruction: {instruction}\n## Response: {response}".format_map(dp)
        # 将序列转换为tokens
        tokens = tokenizer(sequence, return_tensors='pt')['input_ids'].to("cuda")
        # 通过模型获取logits
        with torch.no_grad():
            outputs = model(tokens)
        logits = outputs.logits
        # 对每个token的logits进行处理
        energies = []
        for token_logits in logits[0]:
            # 获取top-k logit的索引
            top_indices = torch.topk(token_logits, top_k).indices
            # 计算top-k logit的softmax
            softmax_values = F.softmax(token_logits[top_indices] / temperature, dim=-1)
            # 计算能量值
            energy = -torch.sum(torch.log(softmax_values)) / temperature
            energies.append(energy.item())
        energy_value = sum(energies) / len(energies)  # 长度是否是个问题？
        energy_dps.append((dp, energy_value))
    # 返回能量值列表
    return energy_dps


def fast_k_center_sampling_from_embedding(data, k):
    n = data.shape[0]
    center = random.randint(0, n)  # Randomly choose the first center
    centers = [center]
    cache_dis = 1 - data[center].unsqueeze(0) @ data.T  # [1, n]
    cache_dis[0, center] = -100

    for _ in range(1, k):
        center = torch.argmax(cache_dis[0])
        centers.append(center)
        new_dis = 1 - data[center].unsqueeze(0) @ data.T
        cache_dis = torch.min(torch.cat([cache_dis, new_dis], 0), 0, keepdim=True)[0]
        cache_dis[0, center] = -100
        if len(centers) % 2000 == 0:
            print(len(centers))
    return centers
