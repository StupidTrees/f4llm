
import copy
from abc import ABC

import torch
import torch.nn as nn
from peft import (TaskType, get_peft_model, prepare_model_for_kbit_training,
                  LoraConfig, PrefixTuningConfig)
from transformers import AutoConfig, BitsAndBytesConfig, AutoModelForCausalLM

from utils.general import is_petuning
from utils.general import get_parameter_number
from utils.register import registry


class BaseModels(ABC):
    def __init__(self, task_name):
        super().__init__()

        self.task_name = task_name
        config = registry.get("config")
        self.model_config = config.model_config
        self.train_config = config.training_config
        self.role = config.F.role
        self.logger = registry.get("logger")
        self.auto_config = self._build_config()

    def _build_config(self):
        auto_config = AutoConfig.from_pretrained(
            self.model_config.model_name_or_path,
            trust_remote_code=True,
        )
        return auto_config

    def build_model(self):
        backbone = self._add_base_model()

        backbone = self._add_quantize_model(backbone)

        if is_petuning(self.model_config.tuning_type):
            backbone = self._add_delta_model(backbone)

        return backbone

    def _add_base_model(self):
        if self.train_config.load_in_8bit or self.train_config.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=self.train_config.load_in_8bit, load_in_4bit=self.train_config.load_in_4bit
            )
            torch_dtype = torch.bfloat16
        else:
            quantization_config = None
            torch_dtype = None

        if self.role == "server":
            device_map = "cpu"
        else:
            device_map = "auto"

        backbone = AutoModelForCausalLM.from_pretrained(
            self.model_config.model_name_or_path,
            config=self.auto_config,
            trust_remote_code=True,
            quantization_config=quantization_config,
            torch_dtype=torch_dtype,
            device_map=device_map
        )
        return backbone

    def _add_delta_model(self, backbone):
        if is_petuning(self.model_config.tuning_type):
            if "lora" in self.model_config.tuning_type:
                target_modules = getattr(self.model_config, "target_modules", self.target_modules)
                peft_config = LoraConfig(task_type=self.task_type,
                                         r=self.model_config.lora_rank,
                                         lora_alpha=self.model_config.lora_alpha,
                                         lora_dropout=self.model_config.lora_dropout,
                                         target_modules=target_modules)

            elif "prefix" in self.model_config.tuning_type:
                peft_config = PrefixTuningConfig(task_type=self.task_type,
                                                 num_virtual_tokens=self.model_config.num_virtual_tokens)
            else:
                raise NotImplementedError(f"NotImplemented tuning_type: {self.model_config.tuning_type}")
            backbone = get_peft_model(backbone, peft_config)
        else:
            raise NotImplementedError(f"NotImplemented tuning_type: {self.model_config.tuning_type}")

        self.logger.debug(f"Delta Model: {self.model_config.tuning_type}, "
                          f"Parameters: {get_parameter_number(backbone)} M")

        return backbone

    def _add_quantize_model(self, backbone):

        if self.train_config.load_in_8bit or self.train_config.load_in_4bit:
            self.logger.info(f"Quantized to 8bit")
            backbone = prepare_model_for_kbit_training(
                backbone, use_gradient_checkpointing=self.train_config.gradient_checkpointing
            )

        return backbone

    @property
    def task_type(self):
        return TaskType.CAUSAL_LM

    @property
    def target_modules(self):
        return None
