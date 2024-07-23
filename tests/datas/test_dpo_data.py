import random
import sys
import unittest
from copy import copy
from os.path import exists, join

import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from configs import build_config
from datas.dpo_data import DPODataManger, UFBDataManger
from datas.sft_data import preprocess, _tokenize_fn, IGNORE_INDEX, LlaMaGenDataManger
from tests.config import test_model_path, test_output_path
import pickle

test_clients_num = 3


class TestDPOData(unittest.TestCase):

    def _random_slicing(self, dataset, num_clients):
        num_items = int(len(dataset) / num_clients)
        dict_users, all_idxs = {}, [i for i in range(len(dataset))]
        for i in range(num_clients):
            dict_users[i] = list(
                np.random.choice(all_idxs, num_items, replace=False))
            all_idxs = list(set(all_idxs) - set(dict_users[i]))
        return dict_users

    def _generate_test_dpo_data(self):
        if not exists(join(test_output_path, 'test_partition_dpo.pkl')):
            q = "Q:How to hang knives against the wall?"
            a0 = "A:First, you need to find a magnetic strip to hang the knives, then attach the knives to the strip, "
            a1 = "A:First, you need to find a magnetic strip to hang the knives, then attach the knives to the strip, " \
                 "and finally hang the strip on the wall, so that the knives can be hung against the wall, " \
                 "which is convenient for storage, and can also prevent children from touching them, which is safer. "
            data = [{'prompt': q, 'rejected': a0, 'chosen': copy(a1)[:random.randint(1, len(a1) // 2)]} for _ in
                    range(100)]
            obj = {'attribute': {'clients_num': test_clients_num}}
            partition_indexes = self._random_slicing(data, test_clients_num)
            obj['train'] = partition_indexes
            obj['test'] = partition_indexes
            obj['valid'] = partition_indexes
            raw_data = {'train': data, 'test': data, 'valid': data}
            with open(join(test_output_path, 'test_raw_dpo.pkl'), 'wb') as f:
                pickle.dump(raw_data, f)
            with open(join(test_output_path, 'test_partition_dpo.pkl'), 'wb') as f:
                pickle.dump(obj, f)

    def setUp(self) -> None:
        sys.argv.extend(
            ["--model_name_or_path", test_model_path,
             "--output_dir", test_output_path,
             "--task_name", "",
             "--overwrite_cache", "true",
             "--clients_num", f"{test_clients_num}",
             "--raw_dataset_path", "",
             "--partition_dataset_path", "",
             "--model_type", "llama2-base",
             "--task_name", "medi",
             "--partition_dataset_path", join(test_output_path, 'test_partition_dpo.pkl'),
             "--raw_dataset_path", join(test_output_path, 'test_raw_dpo.pkl'),
             "--checkpoint_file", test_output_path])
        if 'discover' in sys.argv:
            sys.argv.remove('discover')
        build_config()

        self.model_max_length = 24
        self.tokenizer = AutoTokenizer.from_pretrained(
            test_model_path,
            trust_remote_code=True,
            use_fast=False,
            model_max_length=self.model_max_length
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self._generate_test_dpo_data()

    def test_ufb_data_manager(self):
        dm = UFBDataManger()
        with open(join(test_output_path, 'test_partition_dpo.pkl'), 'rb') as f:
            partition_obj = pickle.load(f)
        for cid in range(test_clients_num):
            train_dataset = dm.train_dataset_dict[cid]
            self.assertEqual(len(train_dataset), len(partition_obj['train'][cid]))


if __name__ == '__main__':
    unittest.main(verbosity=2)
