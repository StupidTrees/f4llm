"""define evaluation scripts for FNLP """

import torch
import numpy as np
from abc import ABC
from collections import defaultdict

from utils.register import registry


class BaseEval(ABC):
    def __init__(self, device, metric):
        self.device = device
        self.metric = metric
        self.task_name = metric.task_name
        self.debug = registry.get("debug")
        self.logger = registry.get("logger")

    def test_and_eval(self, valid_dl, model, model_type, model_output_mode):
        raise NotImplementedError
