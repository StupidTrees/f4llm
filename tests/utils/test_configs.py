import argparse
import unittest

from configs.peft_configs import get_delta_key
from utils.configs import parse_args, build_configs, ds_configs


class TestConfigs(unittest.TestCase):
    def test_parse_args(self):
        args = parse_args()
        self.assertIsInstance(args, argparse.Namespace)

    def test_build_configs(self):
        with self.assertRaises(TypeError):
            args = build_configs()

    def test_ds_configs(self):
        args = parse_args()
        deepspeed_configs = ds_configs(args)
        self.assertIsInstance(deepspeed_configs, dict)

    def test_get_delta_key(self):
        delta_keys = [
            "fine-tuning",
            "prefix",
            "bitfit",
            "lora",
            "adapter",
            "emulator"
        ]
        for dk in delta_keys:
            a, b = get_delta_key(dk)
            self.assertIsNotNone(a)
            self.assertIsNotNone(b)


if __name__ == '__main__':
    unittest.main()
