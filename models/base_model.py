
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
from models.miscs import build_selector, build_emulator, get_layer_module_name


class BaseModels(ABC):
    def __init__(self, task_name):
        super().__init__()

        self.task_name = task_name
        config = registry.get("config")
        self.model_config = config.model_config
        self.train_config = config.training_config
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

        if getattr(self.model_config, "nas_type", None):
            backbone = self._add_nas_model(backbone)

        if is_petuning(self.model_config.tuning_type):
            backbone = self._add_delta_model(backbone)

        # backbone = self._add_quantize_model(backbone)

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

        backbone = AutoModelForCausalLM.from_pretrained(
            self.model_config.model_name_or_path,
            config=self.auto_config,
            trust_remote_code=True,
            quantization_config=quantization_config,
            torch_dtype=torch_dtype,
        )
        return backbone

    def _add_nas_model(self, backbone):

        if self.model_config.nas_type == "selector":
            backbone_copy = build_selector(backbone, self.model_config)
        elif self.model_config.nas_type == "emulator":
            backbone_copy = build_emulator(backbone, self.model_config)
        else:
            raise ValueError(f"{self.model_config.nas_type} is not implemented")

        self.logger.debug(f"Base Model: {self.model_config.nas_type} "
                          f"original model's size: {get_parameter_number(backbone)} | "
                          f"new model's size: {get_parameter_number(backbone_copy)}")

        return backbone_copy

    def _add_delta_model(self, backbone):

        if is_petuning(self.model_config.tuning_type):
            # opendelta discard
            # delta_args = registry.get("delta_config")
            # delta_config = AutoDeltaConfig.from_dict(delta_args)
            # delta_model = AutoDeltaModel.from_config(delta_config, backbone_model=backbone)
            # delta_model.freeze_module(set_state_dict=True)
            # delta_model.log(delta_ratio=True, trainable_ratio=True, visualization=True)
            # self.logger.debug(delta_config)

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
            # backbone.save_pretrained()

        elif self.model_config.tuning_type == "emulator":
            layer_name = get_layer_module_name(backbone=backbone)
            layers = backbone.get_submodule(layer_name)
            freeze_layer = self.model_config.base_layer
            for param in layers[freeze_layer:-freeze_layer].parameters():
                param.requires_grad = False

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

        # if self.model_config.quantization_bit is not None:
        #     self.logger.info(f"Quantized to {self.model_config.quantization_bit} bit")
        #     backbone = backbone.quantize(self.model_config.quantization_bit)
        # # elif registry.get("config").training_config.fp16:
        # #     self.logger.warning(f"Using fp16")  # 不兼容deepspeed
        # elif is_petuning(self.model_config.tuning_type):
        #     # half training
        #     backbone = backbone.half()
        #     for _, param in backbone.named_parameters():
        #         if param.requires_grad:
        #             param.data = param.data.float()

        return backbone

    @property
    def task_type(self):
        return TaskType.CAUSAL_LM

    @property
    def target_modules(self):
        return None
