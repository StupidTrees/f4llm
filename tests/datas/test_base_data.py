import sys
import unittest

import torch

from configs import build_config
from datas.base_data import FedBaseDataset
from tests.config import test_output_path, test_model_path


class TestBaseData(unittest.TestCase):
    def setUp(self) -> None:
        sys.argv.extend(
            ["--model_name_or_path", test_model_path,
             "--output_dir", test_output_path,
             "--task_name", "",
             "--raw_dataset_path", "",
             "--partition_dataset_path", "",
             "--checkpoint_file", test_output_path])
        if 'discover' in sys.argv:
            sys.argv.remove('discover')
        build_config()

    def test_fed_base_dataset(self):
        dummy_features = torch.randn((120, 4))
        dataset = FedBaseDataset(dummy_features)
        for idx, sample in enumerate(dataset.select(5)):
            self.assertTrue((sample == dummy_features[idx]).all())

if __name__ == '__main__':
    unittest.main()
