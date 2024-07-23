import random
import unittest
from copy import deepcopy

import accelerate
from sympy.physics.units import Da
from torch.nn import ModuleList, DataParallel
from torch.nn.parallel import DistributedDataParallel
from transformers import OPTForCausalLM, OPTConfig, GPT2LMHeadModel, GPT2Config, BloomForCausalLM, BloomConfig
from transformers.models.bloom.modeling_bloom import BloomBlock
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.models.opt.modeling_opt import OPTDecoderLayer

from configs import ModelArguments
from models.miscs import get_layer_module_name, get_layer_module, set_layer_module, build_selector, build_emulator, \
    uniform_layer_idx, extract_backbone


class TestMiscs(unittest.TestCase):

    def setUp(self) -> None:
        self.layer_num = 1
        self.test_model_clz = [OPTForCausalLM, GPT2LMHeadModel, BloomForCausalLM]
        self.test_model_cfg_clz = [OPTConfig, GPT2Config, BloomConfig]
        self.test_model_num_layer_name = ["num_hidden_layers", "n_layer", "n_layer"]
        self.test_model_layer_class = [OPTDecoderLayer, GPT2Block, BloomBlock]
        self.test_models = self._create_test_models(self.layer_num)

    def _create_test_models(self, layer_num):
        res = []
        for clz, cfg_clz, num_layer_name in zip(self.test_model_clz, self.test_model_cfg_clz,
                                                self.test_model_num_layer_name):
            res.append(clz(config=cfg_clz(**{num_layer_name: layer_num})))
        return res

    def test_get_layer_module_name(self):
        gts = ["model.decoder.layers", "transformer.h", "transformer.h"]
        for md, gt in zip(self.test_models, gts):
            self.assertEqual(get_layer_module_name(md), gt)

    def test_get_layer_module(self):
        for md, gt in zip(self.test_models, self.test_model_layer_class):
            module, name = get_layer_module(md)
            while isinstance(module, ModuleList):
                module = module[0]
            self.assertIsInstance(module, gt)

    def test_set_layer_module(self):
        test_modules_cpy = deepcopy(self.test_models)
        sub_module = ModuleList()
        for md in test_modules_cpy:
            set_layer_module(md, sub_module)
            self.assertEqual(sub_module, get_layer_module(md)[0])

    def test_build_selector(self):
        layer_num = 10
        test_models = self._create_test_models(layer_num)
        for md, layer_clz in zip(test_models, self.test_model_layer_class):
            md_cfg = ModelArguments("")
            md_cfg.client_model_layers = 0
            for num in [0, 4, layer_num]:
                md_cfg.server_model_layers = random.sample(range(layer_num), num)
                selector = build_selector(md, md_cfg)
                selector_layers, _ = get_layer_module(selector)
                self.assertEqual(len(selector_layers), num)
                if num > 0:
                    self.assertIsInstance(selector_layers[0], layer_clz)
                for i in range(num):
                    self.assertEqual(get_layer_module(selector)[0][i],
                                     get_layer_module(md)[0][md_cfg.server_model_layers[i]])

    def test_uniform_layer_idx(self):
        total_layer = 100
        for server_layer in range(0, 60, 10):
            for base_layer in range(1, total_layer + 1, 10):
                emulator_layer = server_layer - 2 * base_layer
                if emulator_layer < 0:
                    continue
                idx = uniform_layer_idx(total_layer, base_layer, emulator_layer)
                self.assertEqual(len(idx), 2 * base_layer + emulator_layer)

    def test_extract_backbone(self):
        for lm in self.test_models:
            lm_dp = DataParallel(lm)
            backbone = extract_backbone(lm_dp)
            self.assertEqual(backbone, lm)

    def test_build_emulator(self):
        md_cfg = ModelArguments("")
        md_cfg.client_model_layers = []
        self.assertRaises(TypeError, build_emulator, self.test_models[0], md_cfg)
        layer_num = 10
        test_models = self._create_test_models(layer_num)
        for md, layer_clz in zip(test_models, self.test_model_layer_class):
            md_cfg = ModelArguments("")
            md_cfg.client_model_layers = 0
            for server_model_layer in range(1, layer_num + 1):
                for base_layer in range(1, layer_num // 2 + 1):
                    md_cfg.server_model_layers = server_model_layer
                    md_cfg.base_layer = base_layer
                    emulator_layer = server_model_layer - 2 * base_layer
                    if emulator_layer <= 1:
                        continue
                    selected_indexes = uniform_layer_idx(len(get_layer_module(md)[0]), base_layer, emulator_layer)
                    emulator = build_emulator(md, md_cfg)
                    emulator_layers = get_layer_module(emulator)[0]
                    self.assertEqual(len(emulator_layers), 2 * base_layer + emulator_layer)
                    for i, si in enumerate(selected_indexes):
                        self.assertEqual(emulator_layers[i], get_layer_module(md)[0][si])


if __name__ == '__main__':
    unittest.main(verbosity=2)
