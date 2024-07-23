import sys
import unittest
from unittest.mock import patch

import torch

from configs import build_config
from datas.base_data import FedBaseDataset
from tests.config import test_output_path, test_model_path


class TestBaseData(unittest.TestCase):
    def setUp(self) -> None:
        arg_list = sys.argv + ["--model_name_or_path", test_model_path,
                               "--output_dir", test_output_path,
                               "--task_name", "",
                               "--raw_dataset_path", "",
                               "--partition_dataset_path", "",
                               "--checkpoint_file", test_output_path]
        if 'discover' in arg_list:
            arg_list.remove('discover')
        with patch("sys.argv", arg_list):
            build_config()

    def test_fed_base_dataset(self):
        dummy_features = torch.randn((120, 4))
        dataset = FedBaseDataset(dummy_features)
        self.assertEqual(len(dataset), 120)
        for idx, sample in enumerate(dataset.select(5)):
            self.assertTrue((sample == dummy_features[idx]).all())


if __name__ == '__main__':
    unittest.main()
