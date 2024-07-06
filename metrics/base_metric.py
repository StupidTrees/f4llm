
import numpy as np
from abc import ABC
from utils.register import registry


class BaseMetric(ABC):
    def __init__(self, tokenizer, is_decreased_valid_metric=False, save_outputs=False):
        super().__init__()

        self.tokenizer = tokenizer
        self.is_decreased_valid_metric = is_decreased_valid_metric
        self.best_valid_metric = float("inf") if self.is_decreased_valid_metric else -float("inf")
        self.results = {}
        self.best = False
        self.save_outputs = save_outputs
        self.logger = registry.get("logger")

    def calculate_metric(self, *args):
        raise NotImplementedError

    def update_metrics(self, *args):
        raise NotImplementedError

    @property
    def is_best(self):
        return self.best

    @property
    def best_metric(self):
        return self.results

    @property
    def metric_name(self):
        raise NotImplementedError

    def decoded(self, tensor):
        tensor = np.where(tensor != -100, tensor, self.tokenizer.pad_token_id)
        decoded_tensor = self.tokenizer.batch_decode(tensor, skip_special_tokens=True)
        return decoded_tensor

