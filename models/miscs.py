
import copy
from loguru import logger
from omegaconf.listconfig import ListConfig

import torch
import torch.nn as nn
from transformers import (
    OPTForCausalLM,
    GPT2LMHeadModel,
    BloomForCausalLM,
)

PromptType = ["p-tuning", "soft-prompt", "prompt"]


def uniform_layer_idx(total_layer, base_layer, emulator_layer):
    if total_layer < (emulator_layer+2*base_layer):
        raise ValueError(f"model's layer_num needs >= {(emulator_layer+2*base_layer)}")

    last_layer = total_layer - base_layer
    all_layer_idx = [i for i in range(total_layer)]

    submodel_layer_idx = all_layer_idx[0:base_layer]
    sub_layer_idx = all_layer_idx[base_layer:-base_layer]
    stride = (len(sub_layer_idx) - 1) / (emulator_layer - 1)

    for i in range(emulator_layer):
        idx = round(i * stride)
        submodel_layer_idx.append(sub_layer_idx[idx])

    submodel_layer_idx.extend(all_layer_idx[last_layer:])

    return submodel_layer_idx


def get_layer_module(backbone):
    layer_name = get_layer_module_name(backbone)
    layer_module = backbone.get_submodule(layer_name)
    return layer_module, layer_name


def get_layer_module_name(backbone):
    # TODO: more general
    if isinstance(backbone, OPTForCausalLM):
        return "model.decoder.layers"
    elif isinstance(backbone, GPT2LMHeadModel):
        return "transformer.h"
    elif isinstance(backbone, BloomForCausalLM):
        return "transformer.h"
    else:
        raise NotImplementedError


def set_layer_module(backbone, layer_module):
    if isinstance(backbone, OPTForCausalLM):
        backbone.model.decoder.layers = layer_module
    elif isinstance(backbone, GPT2LMHeadModel):
        backbone.transformer.h = layer_module
    elif isinstance(backbone, BloomForCausalLM):
        backbone.transformer.h = layer_module
    else:
        raise NotImplementedError
    return backbone


def build_selector(backbone, model_config):
    if isinstance(model_config.client_model_layers, int):
        model_config.client_model_layers = [i for i in range(model_config.client_model_layers)]

    if isinstance(model_config.server_model_layers, int):
        model_config.server_model_layers = [i for i in range(model_config.server_model_layers)]

    layer_module_name = get_layer_module_name(backbone)
    layer_module = backbone.get_submodule(layer_module_name)

    # Create a copy of the model, modify it with the new list, and return
    backbone_copy = copy.deepcopy(backbone)
    sub_modules = torch.nn.ModuleList()

    # build selected layers
    selected_layer = model_config.server_model_layers

    logger.warning(f"selected layer is {selected_layer}")
    for i in selected_layer:
        sub_modules.append(layer_module[i])
    backbone_copy = set_layer_module(backbone_copy, sub_modules)

    return backbone_copy


def build_emulator(backbone, model_config):

    if not isinstance(model_config.client_model_layers, int) or \
            not isinstance(model_config.server_model_layers, int):
        raise TypeError(f"client/server_model_layers needs int")

    layer_module_name = get_layer_module_name(backbone)
    layer_module = backbone.get_submodule(layer_module_name)

    # Create a copy of the model, modify it with the new list, and return
    backbone_copy = copy.deepcopy(backbone)
    sub_modules = torch.nn.ModuleList()

    # build emulator
    emulator_layer = model_config.server_model_layers - 2 * model_config.base_layer

    selected_layer = uniform_layer_idx(len(layer_module), model_config.base_layer, emulator_layer)

    logger.warning(f"selected layer is {selected_layer}")
    for i in selected_layer:
        sub_modules.append(layer_module[i])
    backbone_copy = set_layer_module(backbone_copy, sub_modules)

    return backbone_copy


def extract_backbone(model):
    if isinstance(model, nn.DataParallel):
        return model.module
    elif isinstance(model, nn.parallel.distributed.DistributedDataParallel):
        return model.module
    return model.backbone
