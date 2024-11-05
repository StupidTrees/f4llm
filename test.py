from datasets import load_dataset
import json

# # 1. 加载数据集
# dataset = load_dataset('amphora/mprom-culturebank', split='train')  # 例如，加载SQuAD数据集的训练集
#
# # 2. 将数据集保存为JSON文件
# dataset.to_json('/userhome/jp/fedsl/data/cb_dpo.json', orient='records', lines=True)


dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF")
# 2. 将数据集保存为JSON文件
dataset.to_json('/userhome/jp/fedsl/data/safe_rlhf.json', orient='records', lines=True)
