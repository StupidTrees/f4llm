import torch
from utils.general import get_peft_parameters
from copy import deepcopy

"""
This module provides tools for serializing and deserializing PyTorch models and their gradients. It includes functionalities to convert model parameters and gradients into a single tensor for easy transmission or storage, and vice versa. The module supports different modes of deserialization to either copy or add the serialized parameters to the existing model parameters. Additionally, it offers specialized serialization and deserialization functions for models with parameters tuned using PEFT (Parameter Efficient Fine-Tuning) techniques.

Functions:
    serialize_model_gradients(model): Serializes the gradients of a PyTorch model into a single tensor.
    serialize_model(model): Serializes the parameters of a PyTorch model into a single tensor.
    deserialize_model(model, serialized_parameters, mode): Deserializes the given tensor into the model's parameters.
    serialize_peft_model(model, tuning_type): Serializes the parameters of a PEFT-tuned PyTorch model into a single tensor.
    deserialize_peft_model(model, serialized_parameters, tuning_type, mode): Deserializes the given tensor into the PEFT-tuned model's parameters.

The module is designed to facilitate the handling of model parameters and gradients in distributed training scenarios, model checkpointing, and parameter-efficient transfer learning methods.
"""


class SerializationTool(object):
    """
    This class provides tools for serializing and deserializing PyTorch models and their gradients.
    """

    @staticmethod
    def serialize_model_gradients(model: torch.nn.Module) -> torch.Tensor:
        """
        Serializes the gradients of a PyTorch model into a single tensor.

        Args:
            model: The PyTorch model whose gradients are to be serialized.

        Returns:
            A tensor containing the serialized gradients of the model.

        """
        gradients = [param.grad.data.view(-1) for param in model.parameters()]
        m_gradients = torch.cat(gradients)
        m_gradients = m_gradients.cpu()
        return m_gradients

    @staticmethod
    def serialize_model(model: torch.nn.Module) -> torch.Tensor:
        """
        Serializes the parameters of a PyTorch model into a single tensor.

        Args:
            model: The PyTorch model whose parameters are to be serialized.

        Returns:
            A tensor containing the serialized parameters of the model.

        """

        parameters = [param.data.view(-1) for param in model.parameters()]
        m_parameters = torch.cat(parameters)
        m_parameters = m_parameters.cpu()

        return m_parameters

    @staticmethod
    def deserialize_model(model: torch.nn.Module,
                          serialized_parameters: torch.Tensor,
                          mode="copy"):
        """
        Assigns serialized parameters to model.parameters. This is done by iterating through ``model.parameters()``
        and assigning the relevant params in ``grad_update``.

        Notes: this function manipulates ``model.parameters``.

        Args:
            model (torch.nn.Module): The PyTorch model whose parameters are to be deserialized.
            serialized_parameters (torch.Tensor): The tensor containing the serialized parameters.
            mode (str): The mode of deserialization. "copy" replaces the parameters with the serialized values, while "add" adds the serialized values to the existing parameters.

        Raises:
            ValueError: if mode is not "copy" or "add".

        Returns:
            None
        """

        current_index = 0  # keep track of where to read from grad_update
        for parameter in model.parameters():
            numel = parameter.data.numel()
            size = parameter.data.size()
            if mode == "copy":
                parameter.data.copy_(
                    serialized_parameters[current_index:current_index + numel].view(size))
            elif mode == "add":
                parameter.data.add_(
                    serialized_parameters[current_index:current_index + numel].view(size))
            else:
                raise ValueError(
                    "Invalid deserialize mode {}, require \"copy\" or \"add\" "
                    .format(mode))
            current_index += numel

    @staticmethod
    def serialize_peft_model(model: torch.nn.Module, tuning_type: str) -> torch.Tensor:
        """
        Serializes the parameters of a PEFT-tuned PyTorch model into a single tensor.

        Args:
            model: The PyTorch model whose parameters are to be serialized.
            tuning_type: The type of PEFT tuning applied to the model.

        Returns:
            A tensor containing the serialized parameters of the PEFT-tuned model

        """

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
        """
        Assigns serialized parameters to PEFT-tuned model.parameters. This is done by iterating through the PEFT
        parameters and assigning the relevant params in ``grad_update``.

        Args:
            model: The PyTorch model whose parameters are to be deserialized using PEFT.
            serialized_parameters: The tensor containing the serialized parameters.
            tuning_type: The type of PEFT tuning applied to the model.
            mode: The mode of deserialization. "copy" replaces the parameters with the serialized values, while "add" adds the serialized values to the existing parameters.

        Raises:
            ValueError: if mode is not "copy" or "add".

        Returns:
            None

        """

        current_index = 0  # keep track of where to read from grad_update
        peft_model_state_dict = get_peft_parameters(model, tuning_type)

        for name, parameter in peft_model_state_dict.items():
            numel = parameter.data.numel()
            size = parameter.data.size()
            if mode == "copy":
                parameter.data.copy_(
                    serialized_parameters[current_index:current_index + numel].view(size))
            elif mode == "add":
                parameter.data.add_(
                    serialized_parameters[current_index:current_index + numel].view(size))
            else:
                raise ValueError(
                    "Invalid deserialize mode {}, require \"copy\" or \"add\" ".format(mode))
            current_index += numel
