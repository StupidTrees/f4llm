import copy
import os
import unittest
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from utils.serialization import SerializationTool


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


def _model_params_multiply(model, times):
    for param in model.parameters():
        param.data = param.data * times


class TestSerializationTool(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # DO NOT change the setting below, the model is pretrained on MNIST
        cls.input_size = 784
        cls.hidden_size = 250
        cls.num_classes = 10
        test_path = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(test_path, '../data/nnModel.ckpt')
        cls.model = Net(cls.input_size, cls.hidden_size, cls.num_classes)
        cls.model.load_state_dict(torch.load(model_path))
        cls.peft_config = LoraConfig(target_modules=['fc1'])
        cls.lora_model = get_peft_model(copy.deepcopy(cls.model), cls.peft_config)

    def assertParametersEqual(self, model1, model2):
        for param1, param2 in zip(model1.parameters(), model2.parameters()):
            self.assertTrue(torch.equal(param1, param2))

    def assertParametersNotEqual(self, model1, model2):
        flags = []
        for param1, param2 in zip(model1.parameters(), model2.parameters()):
            flags.append(torch.equal(param1, param2))
        self.assertIn(False, flags)  # at least one False in flags

    def test_serialize_model_gradients(self):
        # set gradients with calculation
        batch_size = 100
        sample_data = torch.randn(batch_size, self.input_size)
        output = self.model(sample_data)
        res = output.mean()
        res.backward()
        # serialize gradients
        serialized_grads = SerializationTool.serialize_model_gradients(self.model)
        m_grads = torch.Tensor([0])
        for param in self.model.parameters():
            m_grads = torch.cat((m_grads, param.grad.data.view(-1)))
        m_grads = m_grads[1:]
        self.assertTrue(torch.equal(serialized_grads, m_grads))

    @torch.no_grad()
    def test_serialize_model(self):
        serialized_params = SerializationTool.serialize_model(self.model)
        m_params = torch.Tensor([0])
        for param in self.model.parameters():
            m_params = torch.cat((m_params, param.data.view(-1)))
        m_params = m_params[1:]
        self.assertTrue(torch.equal(serialized_params, m_params))

    @torch.no_grad()
    def test_deserialize_model_copy(self):
        model = Net(self.input_size, self.hidden_size, self.num_classes)
        self.assertParametersNotEqual(self.model, model)
        serialized_params = SerializationTool.serialize_model(self.model)
        SerializationTool.deserialize_model(model, serialized_params, mode='copy')
        self.assertParametersEqual(self.model, model)

    @torch.no_grad()
    def test_deserialize_model_add(self):
        model = Net(self.input_size, self.hidden_size, self.num_classes)
        self.assertParametersNotEqual(self.model, model)
        serialized_params = SerializationTool.serialize_model(self.model)
        SerializationTool.deserialize_model(model, serialized_params, mode='copy')  # copy first
        SerializationTool.deserialize_model(model, serialized_params, mode='add')  # add params then
        _model_params_multiply(self.model, 2)  # now self.model.params = self.model.params * 2
        self.assertParametersEqual(self.model, model)

    @torch.no_grad()
    def test_deserialize_model_other(self):
        model = Net(self.input_size, self.hidden_size, self.num_classes)
        serialized_params = SerializationTool.serialize_model(self.model)
        with self.assertRaises(ValueError):
            SerializationTool.deserialize_model(model, serialized_params, mode='minus')

    @torch.no_grad()
    def test_serialize_peft_model_adapter(self):
        serialized_params = SerializationTool.serialize_peft_model(self.model, "adapter")
        m_params = torch.Tensor([0])
        for param in self.model.parameters():
            m_params = torch.cat((m_params, param.data.view(-1)))
        m_params = m_params[1:]
        self.assertTrue(torch.equal(serialized_params, m_params))

    @torch.no_grad()
    def test_serialize_peft_model_other(self):
        serialized_params = SerializationTool.serialize_peft_model(self.lora_model, "other")
        m_params = torch.Tensor([0])
        for nm, param in self.lora_model.named_parameters():
            if 'lora' in nm:
                m_params = torch.cat((m_params, param.data.view(-1)))
        m_params = m_params[1:]
        self.assertTrue(torch.equal(serialized_params, m_params))

    @torch.no_grad()
    def test_deserialize_peft_model_adapter_copy(self):
        model = Net(self.input_size, self.hidden_size, self.num_classes)
        self.assertParametersNotEqual(self.model, model)
        serialized_params = SerializationTool.serialize_peft_model(self.model, "adapter")
        SerializationTool.deserialize_peft_model(model, serialized_params, mode='copy', tuning_type="adapter")
        self.assertParametersEqual(self.model, model)

    @torch.no_grad()
    def test_deserialize_peft_model_adapter_add(self):
        model = Net(self.input_size, self.hidden_size, self.num_classes)
        self.assertParametersNotEqual(self.model, model)
        serialized_params = SerializationTool.serialize_peft_model(self.model, "adapter")
        SerializationTool.deserialize_peft_model(model, serialized_params, mode='copy', tuning_type="adapter")
        SerializationTool.deserialize_peft_model(model, serialized_params, mode='add', tuning_type="adapter")
        _model_params_multiply(self.model, 2)
        self.assertParametersEqual(self.model, model)

    @torch.no_grad()
    def test_deserialize_peft_model_adapter_other(self):
        model = Net(self.input_size, self.hidden_size, self.num_classes)
        model = get_peft_model(model, self.peft_config)
        serialized_params = SerializationTool.serialize_peft_model(self.lora_model, "other")
        with self.assertRaises(ValueError):
            SerializationTool.deserialize_peft_model(model, serialized_params, mode='minus', tuning_type="other")


if __name__ == '__main__':
    unittest.main()
