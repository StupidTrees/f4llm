import os
import pickle
import random
import numpy as np
import pandas as pd

import torch


prompt_cases = {
    1: "现在你是一个法律专家，请你给出下面案件的类型。\n案件：{}\n案件类型是：",
    2: "现在你是一个法律专家，请你给出下面案件的类型。\n案件：{}\n案件类型：",
    3: "请给出下面法律案件的类型。\n案件：{}\n案件类型是：",
    4: "法律案件：{}\n该法律案件类型是：",
    5: "案件：{}\n案件类型是：",
    6: "现在你是一个法律专家，请你判断下面案件的类型。\n案件：{}\n案件类型是：",
    7: "现在你是一个法律专家，请你给出下面民事案件的具体类型。\n案件：{}\n案件类型是：",
    8: "现在你是一个法律专家，请你判断下面民事案件的具体类型。\n案件：{}\n案件类型是：",
    9: "现在你是一个法律专家，请你根据下面的法律分析方式给出下面民事案件的具体类型。法律分析方法：问题，规则，应用，结论。\n案件：{}\n案件类型是："
}

legal_task_prompt = {
    "lcp": "现在你是一个法律专家，请你判断下面案件的类型。\n案件：{}\n案件类型是："

}


def build_legal_prompt(task="lcp"):
    return legal_task_prompt[task]


def build_prompt(prompt, text):
    return prompt.format(text)


def build_two_prompt(prompt, text):
    cot_prompt = '让我们一步一步思考。'
    return prompt.format(text) + "\n{}".format(cot_prompt)


def exact_matching(pre_results, real_results):
    cnt = 0
    correct = 0
    for pre_res, real_res in zip(pre_results, real_results):
        same_res = set(pre_res) & set(real_res)
        if len(same_res) == len(real_res):
            correct += 1
        cnt += 1
    return round(correct / cnt, 3)


def reuse_results(pth, results, batch_size=8):

    skip_batch = -1
    if not os.path.isfile(pth):
        return skip_batch, results

    with open(pth, "rb") as file:
        old_results = pickle.load(file)
    
    # results {prompt:, result: [texts, labels, predict_labels]}
    label, predic = [], []
    for result in old_results["result"]:
        texts, labels, predict_labels = result
        label.append(labels)
        predic.append(predict_labels)
        results["result"].append((texts, labels, predict_labels))
    skip_batch = len(results["result"]) // batch_size
    reuse_num = skip_batch * batch_size
    results["result"] = results["result"][0:reuse_num]
    print(f"load {len(results['result'])} predictions, EM: {exact_matching(predic, label)}")

    return skip_batch, results
