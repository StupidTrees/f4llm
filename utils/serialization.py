
import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict, PeftModelForCausalLM
from transformers.modeling_utils import unwrap_model
from trl import AutoModelForCausalLMWithValueHead

from utils.general import get_peft_parameters
from copy import deepcopy


class SerializationTool(object):
    @staticmethod
    def serialize_model_gradients(model: torch.nn.Module) -> torch.Tensor:
        gradients = [param.grad.data.view(-1) for param in model.parameters()]
        m_gradients = torch.cat(gradients)
        m_gradients = m_gradients.cpu()
        return m_gradients

    @staticmethod
    def serialize_model(model: torch.nn.Module) -> torch.Tensor:
        """Unfold model parameters

        Unfold every layer of model, concate all of tensors into one.
        Return a `torch.Tensor` with shape (size, ).

        Args:
            model (torch.nn.Module): model to serialize.
        """

        parameters = [param.data.view(-1) for param in model.parameters()]
        m_parameters = torch.cat(parameters)
        m_parameters = m_parameters.cpu()

        return m_parameters

    @staticmethod
    def deserialize_model(model: torch.nn.Module,
                          serialized_parameters: torch.Tensor,
                          mode="copy"):
        """Assigns serialized parameters to model.parameters.
        This is done by iterating through ``model.parameters()`` and assigning the relevant params in ``grad_update``.
        NOTE: this function manipulates ``model.parameters``.

        Args:
            model (torch.nn.Module): model to deserialize.
            serialized_parameters (torch.Tensor): serialized model parameters.
            mode (str): deserialize mode. "copy" or "add".
        """

        current_index = 0  # keep track of where to read from grad_update
        for parameter in model.parameters():
            numel = parameter.data.numel()
            size = parameter.data.size()
            if mode == "copy":
                parameter.data.copy_(
                    serialized_parameters[current_index:current_index +
                                                        numel].view(size))
            elif mode == "add":
                parameter.data.add_(
                    serialized_parameters[current_index:current_index +
                                                        numel].view(size))
            else:
                raise ValueError(
                    "Invalid deserialize mode {}, require \"copy\" or \"add\" "
                        .format(mode))
            current_index += numel

    @staticmethod
    def serialize_peft_model(model: torch.nn.Module, tuning_type: str):
        # old version
        # peft_model_state_dict = get_peft_parameters(model, tuning_type)
        # parameters = [param.data.view(-1) for state_dict, param in peft_model_state_dict.items()]
        # # parameters = [parameter.to("cuda") for parameter in parameters]
        # m_parameters = torch.cat(parameters)
        # m_parameters = deepcopy(m_parameters.cpu())

        if isinstance(unwrap_model(model), AutoModelForCausalLMWithValueHead):
            m_parameters = get_peft_model_state_dict(unwrap_model(model).pretrained_model)
            v_head_state_dict = model.v_head.state_dict()
            for k, v in v_head_state_dict.items():
                m_parameters[f"v_head.{k}"] = v

        elif isinstance(unwrap_model(model), PeftModelForCausalLM):
            m_parameters = get_peft_model_state_dict(model)

        else:
            raise TypeError(f"Our peft serialize support PeftModelForCausalLM or AutoModelForCausalLMWithValueHead, "
                            f"but find {type(model)}")

        return m_parameters

    @staticmethod
    def deserialize_peft_model(model: torch.nn.Module,
                               serialized_parameters,
                               tuning_type: str,
                               mode="copy"):
        # old version
        # current_index = 0  # keep track of where to read from grad_update
        # peft_model_state_dict = get_peft_parameters(model, tuning_type)
        #
        # for name, parameter in peft_model_state_dict.items():
        #     numel = parameter.data.numel()
        #     size = parameter.data.size()
        #     if mode == "copy":
        #         parameter.data.copy_(
        #             serialized_parameters[current_index:current_index+numel].view(size))
        #     elif mode == "add":
        #         parameter.data.add_(
        #             serialized_parameters[current_index:current_index+numel].view(size))
        #     else:
        #         raise ValueError(
        #             "Invalid deserialize mode {}, require \"copy\" or \"add\" ".format(mode))
        #     current_index += numel

        if isinstance(unwrap_model(model), AutoModelForCausalLMWithValueHead):
            set_peft_model_state_dict(unwrap_model(model).pretrained_model, serialized_parameters)
            model.v_head.load_state_dict({
                "summary.weight": serialized_parameters["v_head.summary.weight"],
                "summary.bias": serialized_parameters["v_head.summary.bias"]
            })

        elif isinstance(unwrap_model(model), PeftModelForCausalLM):
            set_peft_model_state_dict(model, serialized_parameters)

        else:
            raise TypeError(f"Our peft deserialize tool support PeftModelForCausalLM or "
                            f"AutoModelForCausalLMWithValueHead, but find {type(model)}")
