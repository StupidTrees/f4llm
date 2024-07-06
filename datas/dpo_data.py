
import random
import numpy as np
from utils.register import registry
from datas.base_data import FedBaseDataManger, FedBaseDataset

import torch
from datasets import Dataset


@registry.register_data("ultrafeedback_binarized")
class DPODataManger(FedBaseDataManger):
    def __init__(self):
        super().__init__()
        self._load_data()

    def build_inputs(self, prompt_text, text):
        inputs_text = prompt_text.format(text)
        return inputs_text

    def process_examples(self, examples, mode="train", verbose=True):
        instances = []
        for idx, example in enumerate(examples):
            instances.append(
                {"idx": f"{mode}-{idx}", "example": example}
            )
        return instances

    def build_dataset(self, features):
        data = {"prompt": [], "chosen": [], "rejected": []}
        for feature in features:
            example = feature["example"]
            for key in example:
                data[key].append(example[key])
        dataset = Dataset.from_dict(data)
        return dataset

    def coll_fn(self, model):
        return None
