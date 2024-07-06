#
# import re
# import warnings
# from dataclasses import asdict, replace
# from enum import Enum
#
#
# import torch
# import torch.nn as nn
# from tqdm import tqdm
# from transformers.pytorch_utils import Conv1D
#
#
# from peft.tuners.tuners_utils import BaseTuner
# from peft.utils import (
#     CLAMP_QUANTILE,
#     COMMON_LAYERS_PATTERN,
#     TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
#     ModulesToSaveWrapper,
#     _freeze_adapter,
#     _get_submodules,
#     get_auto_gptq_quant_linear,
#     get_quantization_config,
# )
# from peft.tuners.lora import (
#     LoraLayer, Linear, Embedding, Conv2d, LoraConfig,
#     Linear4bit, Linear8bitLt, QuantLinear
# )
# from peft.import_utils import is_bnb_4bit_available, is_bnb_available
#
#
# if is_bnb_available():
#     import bitsandbytes as bnb
#
#
# class MyLoraModel(BaseTuner):
#     def __init__(self, model, config, adapter_name) -> None:
#         super().__init__(model, config, adapter_name)
#
#     def _check_new_adapter_config(self, config: LoraConfig) -> None:
#         """
#         A helper method to check the config when a new adapter is being added.
#
#         Raise a ValueError if there is something wrong with the config or if it conflicts with existing adapters.
#
#         """
#         # TODO: there should be a check if any of the existing adapters actually has bias != "none", or else the check
#         # does not fully correspond to the error message.
#         if (len(self.peft_config) > 1) and (config.bias != "none"):
#             raise ValueError(
#                 f"{self.__class__.__name__} supports only 1 adapter with bias. When using multiple adapters, "
#                 "set bias to 'none' for all adapters."
#             )
#
#     @staticmethod
#     def _check_target_module_exists(lora_config, key):
#         """根据target_module以及需要插入的layer得到该层参数是否添加"""
#         if isinstance(lora_config.target_modules, str):
#             target_module_found = re.fullmatch(lora_config.target_modules, key)
#         else:
#             target_module_found = any(
#                 re.match(f".*\.{target_key}$", key) for target_key in lora_config.target_modules
#             ) or any(target_key == key for target_key in lora_config.target_modules)
#             is_using_layer_indexes = getattr(lora_config, "layers_to_transform", None) is not None
#             layer_indexing_pattern = getattr(lora_config, "layers_pattern", None)
#
#             if is_using_layer_indexes and target_module_found:
#                 layers_pattern = COMMON_LAYERS_PATTERN if layer_indexing_pattern is None else layer_indexing_pattern
#                 layers_pattern = [layers_pattern] if isinstance(layers_pattern, str) else layers_pattern
#
#                 for pattern in layers_pattern:
#                     layer_index = re.match(f".*.{pattern}\.(\d+)\.*", key)
#                     if layer_index is not None:
#                         layer_index = int(layer_index.group(1))
#                         if isinstance(lora_config.layers_to_transform, int):
#                             target_module_found = layer_index == lora_config.layers_to_transform
#                         else:
#                             target_module_found = layer_index in lora_config.layers_to_transform
#
#                         break
#                     else:
#                         target_module_found = False
#         return target_module_found
#
#     def _create_and_replace(
#         self,
#         lora_config,
#         adapter_name,
#         target,
#         target_name,
#         parent,
#         **optionnal_kwargs,
#     ):
#         bias = hasattr(target, "bias") and target.bias is not None
#         kwargs = {"r": lora_config.r, "lora_alpha": lora_config.lora_alpha, "lora_dropout": lora_config.lora_dropout,
#                   "fan_in_fan_out": lora_config.fan_in_fan_out, "init_lora_weights": lora_config.init_lora_weights,
#                   "loaded_in_8bit": optionnal_kwargs.pop("loaded_in_8bit", False),
#                   "loaded_in_4bit": optionnal_kwargs.pop("loaded_in_4bit", False), "bias": bias}
#
#         quantization_config = get_quantization_config(self.model, method="gptq")
#         if quantization_config is not None:
#             kwargs["gptq_quantization_config"] = quantization_config
#
#         # TODO: better deal with that
#         if isinstance(target, LoraLayer) and isinstance(target, torch.nn.Conv2d):
#             target.update_layer_conv2d(
#                 adapter_name,
#                 lora_config.r,
#                 lora_config.lora_alpha,
#                 lora_config.lora_dropout,
#                 lora_config.init_lora_weights,
#             )
#         elif isinstance(target, LoraLayer) and isinstance(target, torch.nn.Embedding):
#             target.update_layer_embedding(
#                 adapter_name,
#                 lora_config.r,
#                 lora_config.lora_alpha,
#                 lora_config.lora_dropout,
#                 lora_config.init_lora_weights,
#             )
#
#         elif isinstance(target, LoraLayer):
#             target.update_layer(
#                 adapter_name,
#                 lora_config.r,
#                 lora_config.lora_alpha,
#                 lora_config.lora_dropout,
#                 lora_config.init_lora_weights,
#             )
#         else:
#             new_module = self._create_new_module(lora_config, adapter_name, target, **kwargs)
#             self._replace_module(parent, target_name, new_module, target)
#
#     @staticmethod
#     def _replace_module(parent, child_name, new_module, child):
#         '''
#         :param parent: 需要添加lora的整个module list
#         :param child_name: target_module_name
#         :param new_module: 新建的lora层 linear
#         :param child: target_module 基本上就是 lora linear
#         :return:
#         '''
#         setattr(parent, child_name, new_module)
#         new_module.weight = child.weight
#         if hasattr(child, "bias"):
#             if child.bias is not None:
#                 new_module.bias = child.bias
#
#         if getattr(child, "state", None) is not None:
#             new_module.state = child.state
#             new_module.to(child.weight.device)
#
#         # dispatch to correct device
#         for name, module in new_module.named_modules():
#             if "lora_" in name:
#                 module.to(child.weight.device)
#             if "ranknum" in name:
#                 module.to(child.weight.device)
#
#     def _mark_only_adapters_as_trainable(self) -> None:
#         active_adapter = self._get_active_adapter()
#         bias = self.peft_config[active_adapter].bias
#
#         for n, p in self.model.named_parameters():
#             if "lora_" not in n:
#                 p.requires_grad = False
#         if bias == "none":
#             return
#         elif bias == "all":
#             for n, p in self.model.named_parameters():
#                 if "bias" in n:
#                     p.requires_grad = True
#         elif bias == "lora_only":
#             for m in self.model.modules():
#                 if isinstance(m, LoraLayer) and hasattr(m, "bias") and m.bias is not None:
#                     m.bias.requires_grad = True
#         else:
#             raise NotImplementedError
#
#     @staticmethod
#     def _create_new_module(lora_config, adapter_name, target, **kwargs):
#         gptq_quantization_config = kwargs.get("gptq_quantization_config", None)
#         AutoGPTQQuantLinear = get_auto_gptq_quant_linear(gptq_quantization_config)
#
#         loaded_in_8bit = kwargs.pop("loaded_in_8bit", False)
#         loaded_in_4bit = kwargs.pop("loaded_in_4bit", False)
#         bias = kwargs.pop("bias", False)
#
#         if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
#             eightbit_kwargs = kwargs.copy()
#             eightbit_kwargs.update(
#                 {
#                     "has_fp16_weights": target.state.has_fp16_weights,
#                     "memory_efficient_backward": target.state.memory_efficient_backward,
#                     "threshold": target.state.threshold,
#                     "index": target.index,
#                 }
#             )
#             new_module = Linear8bitLt(
#                 adapter_name, target.in_features, target.out_features, bias=bias, **eightbit_kwargs
#             )
#         elif loaded_in_4bit and is_bnb_4bit_available() and isinstance(target, bnb.nn.Linear4bit):
#             fourbit_kwargs = kwargs.copy()
#             fourbit_kwargs.update(
#                 {
#                     "compute_dtype": target.compute_dtype,
#                     "compress_statistics": target.weight.compress_statistics,
#                     "quant_type": target.weight.quant_type,
#                 }
#             )
#             new_module = Linear4bit(adapter_name, target.in_features, target.out_features, bias=bias, **fourbit_kwargs)
#         elif AutoGPTQQuantLinear is not None and isinstance(target, AutoGPTQQuantLinear):
#             new_module = QuantLinear(adapter_name, target, **kwargs)
#             target.weight = target.qweight
#         elif isinstance(target, torch.nn.Embedding):
#             embedding_kwargs = kwargs.copy()
#             embedding_kwargs.pop("fan_in_fan_out", None)
#             in_features, out_features = target.num_embeddings, target.embedding_dim
#             new_module = Embedding(adapter_name, in_features, out_features, **embedding_kwargs)
#         elif isinstance(target, torch.nn.Conv2d):
#             out_channels, in_channels = target.weight.size()[:2]
#             kernel_size = target.weight.size()[2:]
#             stride = target.stride
#             padding = target.padding
#             new_module = Conv2d(adapter_name, in_channels, out_channels, kernel_size, stride, padding, **kwargs)
#         else:
#             if isinstance(target, torch.nn.Linear):
#                 in_features, out_features = target.in_features, target.out_features
#                 if kwargs["fan_in_fan_out"]:
#                     warnings.warn(
#                         "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
#                         "Setting fan_in_fan_out to False."
#                     )
#                     kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
#             elif isinstance(target, Conv1D):
#                 in_features, out_features = (
#                     target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
#                 )
#                 kwargs["is_target_conv_1d_layer"] = True
#                 if not kwargs["fan_in_fan_out"]:
#                     warnings.warn(
#                         "fan_in_fan_out is set to False but the target module is `Conv1D`. "
#                         "Setting fan_in_fan_out to True."
#                     )
#                     kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
#             else:
#                 raise ValueError(
#                     f"Target module {target} is not supported. "
#                     f"Currently, only `torch.nn.Linear` and `Conv1D` are supported."
#                 )
#             new_module = Linear(adapter_name, in_features, out_features, bias=bias, **kwargs)
#
#         return new_module
#
#     def __getattr__(self, name: str):
#         """Forward missing attributes to the wrapped module."""
#         try:
#             return super().__getattr__(name)  # defer to nn.Module's logic
#         except AttributeError:
#             return getattr(self.model, name)
#
#     def get_peft_config_as_dict(self, inference: bool = False):
#         config_dict = {}
#         for key, value in self.peft_config.items():
#             config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(value).items()}
#             if inference:
#                 config["inference_mode"] = True
#         config_dict[key] = config
#         return config
#
#     def _set_adapter_layers(self, enabled=True):
#         for module in self.model.modules():
#             if isinstance(module, LoraLayer):
#                 module.disable_adapters = False if enabled else True
#             elif isinstance(module, ModulesToSaveWrapper):
#                 module.disable_adapters = False if enabled else True
#
#     def enable_adapter_layers(self):
#         self._set_adapter_layers(enabled=True)
#
#     def _get_active_adapter(self) -> str:
#         active_adapter = None
#         for module in self.model.modules():
#             if isinstance(module, LoraLayer):
#                 active_adapter = module.active_adapter
#
#         if active_adapter is None:
#             raise ValueError(
#                 "Something went wrong, no active adapter could be found, please report the issue on GitHub"
#             )
#         return active_adapter
#
#     def disable_adapter_layers(self):
#         active_adapter = self._get_active_adapter()
#         val = self.peft_config[active_adapter].bias
#         if val != "none":
#             msg = (
#                 f"Careful, disabling adapter layers with bias configured to be '{val}' does not produce the same "
#                 "output as the the base model would without adaption."
#             )
#             warnings.warn(msg)
#         self._set_adapter_layers(enabled=False)
#
#     def set_adapter(self, adapter_name):
#         for module in self.model.modules():
#             if isinstance(module, LoraLayer):
#                 if module.merged:
#                     warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
#                     module.unmerge()
#                 module.active_adapter = adapter_name
#
#     @staticmethod
#     def _prepare_adapter_config(peft_config, model_config):
#         """自动探测 target_modules"""
#         if peft_config.target_modules is None:
#             if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
#                 raise ValueError("Please specify `target_modules` in `peft_config`")
#             peft_config.target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_config["model_type"]]
#         return peft_config
#
#     def _unload_and_optionally_merge(self, merge=True, progressbar: bool = False):
#         if getattr(self.model, "is_loaded_in_8bit", False) or getattr(self.model, "is_loaded_in_4bit", False):
#             raise ValueError("Cannot merge LORA layers when the model is loaded in 8-bit mode")
#         if getattr(self.model, "quantization_method", None) == "gptq":
#             raise ValueError("Cannot merge LORA layers when the model is gptq quantized")
#
#         key_list = [key for key, _ in self.model.named_modules() if "lora" not in key]
#         desc = "Unloading " + ("and merging " if merge else "") + "model"
#         for key in tqdm(key_list, disable=not progressbar, desc=desc):
#             try:
#                 parent, target, target_name = _get_submodules(self.model, key)
#             except AttributeError:
#                 continue
#             if isinstance(target, LoraLayer):
#                 if isinstance(target, nn.Embedding):
#                     new_module = torch.nn.Embedding(target.in_features, target.out_features)
#                 elif isinstance(target, nn.Conv2d):
#                     new_module = torch.nn.Conv2d(
#                         target.in_channels,
#                         target.out_channels,
#                         kernel_size=target.kernel_size,
#                         stride=target.stride,
#                         padding=target.padding,
#                         dilation=target.dilation,
#                     )
#                 else:
#                     bias = target.bias is not None
#                     if getattr(target, "is_target_conv_1d_layer", False):
#                         new_module = Conv1D(target.out_features, target.in_features)
#                     else:
#                         new_module = torch.nn.Linear(target.in_features, target.out_features, bias=bias)
#                 if merge:
#                     target.merge()
#                 self._replace_module(parent, target_name, new_module, target)
#
#             # save any additional trainable modules part of `modules_to_save`
#             if isinstance(target, ModulesToSaveWrapper):
#                 setattr(parent, target_name, target.modules_to_save[target.active_adapter])
#
#         return self.model
#
#     def add_weighted_adapter(self, adapters, weights, adapter_name, combination_type="svd"):
#         """
#         This method adds a new adapter by merging the given adapters with the given weights.
#
#         Args:
#             adapters (list): List of adapter names to be merged.
#             weights (list): List of weights for each adapter.
#             adapter_name (str): Name of the new adapter.
#             combination_type (str): Type of merging. Can be one of [`svd`, `linear`]
#         """
#
#         if adapter_name in list(self.peft_config.keys()):
#             return
#         for adapter in adapters:
#             if adapter not in list(self.peft_config.keys()):
#                 raise ValueError(f"Adapter {adapter} does not exist")
#
#         # if there is only one adapter, we can only use linear merging
#         combination_type = "linear" if len(adapters) == 1 else combination_type
#
#         # new rank is the max of all ranks of the adapters
#         unique_ranks = list({self.peft_config[adapter].r for adapter in adapters})
#         if combination_type == "linear":
#             if len(unique_ranks) != 1:
#                 raise ValueError("All adapters must have the same r value when using `linear` combination_type")
#             new_rank = unique_ranks[0]
#         elif combination_type == "svd":
#             new_rank = max(unique_ranks)
#         else:
#             raise ValueError(f"Invalid combination_type: {combination_type}")
#
#         self.peft_config[adapter_name] = replace(self.peft_config[adapters[0]], r=new_rank, lora_alpha=new_rank)
#         self.inject_adapter(self.model, adapter_name)
#
#         # Do we really need that?
#         _freeze_adapter(self.model, adapter_name)
#
#         key_list = [key for key, _ in self.model.named_modules() if "lora" not in key]
#         for key in key_list:
#             _, target, _ = _get_submodules(self.model, key)
#             if isinstance(target, LoraLayer):
#                 if adapter_name in target.lora_A:
#                     target_lora_A = target.lora_A[adapter_name].weight
#                     target_lora_B = target.lora_B[adapter_name].weight
#                 elif adapter_name in target.lora_embedding_A:
#                     target_lora_A = target.lora_embedding_A[adapter_name]
#                     target_lora_B = target.lora_embedding_B[adapter_name]
#
#                 target_lora_A.data = target_lora_A.data * 0.0
#                 target_lora_B.data = target_lora_B.data * 0.0
#                 if combination_type == "linear":
#                     for adapter, weight in zip(adapters, weights):
#                         if adapter in target.lora_A:
#                             current_adapter_lora_A = target.lora_A[adapter].weight
#                             current_adapter_lora_B = target.lora_B[adapter].weight
#                         elif adapter in target.lora_embedding_A:
#                             current_adapter_lora_A = target.lora_embedding_A[adapter]
#                             current_adapter_lora_B = target.lora_embedding_B[adapter]
#                         target_lora_A.data += current_adapter_lora_A.data * weight * target.scaling[adapter]
#                         target_lora_B.data += current_adapter_lora_B.data
#                 elif combination_type == "svd":
#                     target_lora_A.data, target_lora_B.data = self._svd_weighted_adapter(
#                         adapters, weights, new_rank, target, target_lora_A, target_lora_B
#                     )
#
#     def _svd_weighted_adapter(self, adapters, weights, new_rank, target, target_lora_A, target_lora_B):
#         delta_weight = weights[0] * target.get_delta_weight(adapters[0])
#         for adapter, weight in zip(adapters[1:], weights[1:]):
#             delta_weight += weight * target.get_delta_weight(adapter)
#         conv2d = isinstance(target, Conv2d)
#         if conv2d:
#             conv2d_1x1 = target.weight.size()[2:4] == (1, 1)
#             if not conv2d_1x1:
#                 delta_weight = delta_weight.flatten(start_dim=1)
#             else:
#                 delta_weight = delta_weight.squeeze()
#         if target.fan_in_fan_out:
#             delta_weight = delta_weight.T
#
#         # based on https://github.com/kohya-ss/sd-scripts/blob/main/networks/svd_merge_lora.py#L114-L131
#         U, S, Vh = torch.linalg.svd(delta_weight)
#         U = U[:, :new_rank]
#         S = S[:new_rank]
#         U = U @ torch.diag(S)
#         Vh = Vh[:new_rank, :]
#         dist = torch.cat([U.flatten(), Vh.flatten()])
#         hi_val = torch.quantile(dist, CLAMP_QUANTILE)
#         low_val = -hi_val
#         U = U.clamp(low_val, hi_val)
#         Vh = Vh.clamp(low_val, hi_val)
#         if conv2d:
#             U = U.reshape(target_lora_B.data.shape)
#             Vh = Vh.reshape(target_lora_A.data.shape)
#         return Vh, U
#
#     def delete_adapter(self, adapter_name):
#         """
#         Deletes an existing adapter.
#
#         Args:
#             adapter_name (str): Name of the adapter to be deleted.
#         """
#         if adapter_name not in list(self.peft_config.keys()):
#             raise ValueError(f"Adapter {adapter_name} does not exist")
#         del self.peft_config[adapter_name]
#         key_list = [key for key, _ in self.model.named_modules() if "lora" not in key]
#         for key in key_list:
#             _, target, _ = _get_submodules(self.model, key)
#             if isinstance(target, LoraLayer):
#                 for attr in [
#                     "r",
#                     "lora_alpha",
#                     "scaling",
#                     "lora_A",
#                     "lora_B",
#                     "lora_embedding_A",
#                     "lora_embedding_B",
#                     "lora_dropout",
#                 ]:
#                     if adapter_name in getattr(target, attr):
#                         getattr(target, attr).pop(adapter_name)
#                 if target.active_adapter == adapter_name:
#                     resetting_active_adapter = list(self.peft_config.keys())[0]
#                     warnings.warn(
#                         f"Adapter {adapter_name} was active which is now deleted. Setting active adapter to {resetting_active_adapter}. "
#                     )
#                     target.active_adapter = resetting_active_adapter
#
#     def merge_and_unload(self, progressbar: bool = False):
#         r"""
#         This method merges the LoRa layers into the base model. This is needed if someone wants to use the base model
#         as a standalone model.
#
#         Args:
#             progressbar (bool): whether to show a progressbar indicating the unload and merge process
#
#         Example:
#
#         ```py
#         >>> from transformers import AutoModelForCausalLM
#         >>> from peft import PeftModel
#
#         >>> base_model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-40b")
#         >>> peft_model_id = "smangrul/falcon-40B-int4-peft-lora-sfttrainer-sample"
#         >>> model = PeftModel.from_pretrained(base_model, peft_model_id)
#         >>> merged_model = model.merge_and_unload()
#         ```
#         """
#         return self._unload_and_optionally_merge(progressbar=progressbar)
#
#     def unload(self):
#         """
#         Gets back the base model by removing all the lora modules without merging. This gives back the original base
#         model.
#         """
#         return self._unload_and_optionally_merge(merge=False)