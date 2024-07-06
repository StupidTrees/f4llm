
import torch
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
    def serialize_peft_model(model: torch.nn.Module, tuning_type: str) -> torch.Tensor:

        peft_model_state_dict = get_peft_parameters(model, tuning_type)
        parameters = [param.data.view(-1) for state_dict, param in peft_model_state_dict.items()]
        # parameters = [parameter.to("cuda") for parameter in parameters]
        m_parameters = torch.cat(parameters)
        m_parameters = deepcopy(m_parameters.cpu())

        return m_parameters

    @staticmethod
    def deserialize_peft_model(model: torch.nn.Module,
                               serialized_parameters: torch.Tensor,
                               tuning_type: str,
                               mode="copy"):

        current_index = 0  # keep track of where to read from grad_update
        peft_model_state_dict = get_peft_parameters(model, tuning_type)

        for name, parameter in peft_model_state_dict.items():
            numel = parameter.data.numel()
            size = parameter.data.size()
            if mode == "copy":
                parameter.data.copy_(
                    serialized_parameters[current_index:current_index+numel].view(size))
            elif mode == "add":
                parameter.data.add_(
                    serialized_parameters[current_index:current_index+numel].view(size))
            else:
                raise ValueError(
                    "Invalid deserialize mode {}, require \"copy\" or \"add\" ".format(mode))
            current_index += numel
