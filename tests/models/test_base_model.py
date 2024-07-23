import random
import sys
import unittest

from configs import ModelArguments, build_config
from models.base_model import BaseModels
from models.miscs import get_layer_module

test_model_path = "/data/stupidtree/data/sfl/models/gpt2-large"
test_output_path = "/data/stupidtree/project/f4llm/output"


class TestBaseModel(unittest.TestCase):

    def setUp(self) -> None:
        sys.argv.extend(
            ["--model_name_or_path", test_model_path,
             "--output_dir", test_output_path,
             "--task_name", "",
             "--raw_dataset_path", "",
             "--partition_dataset_path", "",
             "--checkpoint_file", test_output_path,
             "--load_in_4bit", "True"])
        if 'discover' in sys.argv:
            sys.argv.remove('discover')
        build_config()
        self.test_model = BaseModels(None)
        self.raw_backbone_size = len(get_layer_module(self.test_model._add_base_model())[0])

    def test_lora(self):
        model_config = ModelArguments(test_model_path)
        model_config.tuning_type = 'lora'
        model_config.target_modules = ['c_proj']
        self.test_model.model_config = model_config
        backbone = self.test_model.build_model()
        for nm, p in backbone.named_parameters():
            if p.requires_grad:
                self.assertTrue('lora' in nm)

    def test_prefix(self):
        model_config = ModelArguments(test_model_path)
        model_config.tuning_type = 'prefix'
        model_config.num_virtual_tokens = 13
        self.test_model.model_config = model_config
        backbone = self.test_model.build_model()
        for nm, p in backbone.named_parameters():
            if p.requires_grad:
                self.assertTrue('prompt_encoder' in nm)

    def test_emulator(self):
        model_config = ModelArguments(test_model_path)
        model_config.nas_type = 'emulator'
        model_config.tuning_type = ''
        model_config.client_model_layers = 0
        model_config.base_layer = 5
        for server_layer_num in [self.raw_backbone_size // 2, self.raw_backbone_size]:
            model_config.server_model_layers = server_layer_num
            self.test_model.model_config = model_config
            backbone = self.test_model.build_model()
            layers, _ = get_layer_module(backbone)
            self.assertEqual(len(layers), server_layer_num)

    def test_selector(self):
        model_config = ModelArguments(test_model_path)
        model_config.nas_type = 'selector'
        model_config.tuning_type = ''
        model_config.client_model_layers = 0
        model_config.base_layer = 5
        for selected_num in range(0, self.raw_backbone_size + 1, 8):
            model_config.server_model_layers = random.sample(range(self.raw_backbone_size),
                                                             selected_num)
            self.test_model.model_config = model_config
            backbone = self.test_model.build_model()
            layers, _ = get_layer_module(backbone)
            self.assertEqual(len(layers), selected_num)


if __name__ == '__main__':
    unittest.main()
