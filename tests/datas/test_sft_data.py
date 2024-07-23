import pickle
import random
import sys
import unittest
from copy import copy
from os.path import exists, join

import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from configs import build_config
from datas.dpo_data import DPODataManger
from datas.sft_data import preprocess, _tokenize_fn, IGNORE_INDEX, LlaMaGenDataManger
from tests.config import test_model_path, test_output_path

test_clients_num = 3


class TestSFTData(unittest.TestCase):

    def _random_slicing(self, dataset, num_clients):
        num_items = int(len(dataset) / num_clients)
        dict_users, all_idxs = {}, [i for i in range(len(dataset))]
        for i in range(num_clients):
            dict_users[i] = list(
                np.random.choice(all_idxs, num_items, replace=False))
            all_idxs = list(set(all_idxs) - set(dict_users[i]))
        return dict_users

    def _generate_test_partition_data(self):
        if not exists(join(test_output_path, 'test_partition.pkl')):
            q = "Q:How to hang knives against the wall?"
            a = "A:First, you need to find a magnetic strip to hang the knives, then attach the knives to the strip, " \
                "and finally hang the strip on the wall, so that the knives can be hung against the wall, " \
                "which is convenient for storage, and can also prevent children from touching them, which is safer. "
            data = [{'instruction': q, 'response': copy(a)[:random.randint(1, len(a) // 2)]} for _ in range(100)]
            obj = {'attribute': {'clients_num': test_clients_num}}
            partition_indexes = self._random_slicing(data, test_clients_num)
            obj['train'] = partition_indexes
            obj['test'] = partition_indexes
            obj['valid'] = partition_indexes
            raw_data = {'train': data, 'test': data, 'valid': data}
            with open(join(test_output_path, 'test_raw.pkl'), 'wb') as f:
                pickle.dump(raw_data, f)
            with open(join(test_output_path, 'test_partition.pkl'), 'wb') as f:
                pickle.dump(obj, f)


    def setUp(self) -> None:
        sys.argv.extend(
            ["--model_name_or_path", test_model_path,
             "--output_dir", test_output_path,
             "--task_name", "",
             "--clients_num", f"{test_clients_num}",
             "--raw_dataset_path", "",
             "--overwrite_cache", "true",
             "--partition_dataset_path", "",
             "--model_type", "llama2-base",
             "--task_name", "medi",
             "--partition_dataset_path", join(test_output_path, 'test_partition.pkl'),
             "--raw_dataset_path", join(test_output_path, 'test_raw.pkl'),
             "--checkpoint_file", test_output_path])
        if 'discover' in sys.argv:
            sys.argv.remove('discover')
        build_config()

        self.model_max_length = 24
        self.test_sources = ["Q:How to hang knives against the wall?", "Q:Why does the sun rise in the east?"]
        self.test_targets = [
            "A:First, you need to find a magnetic strip to hang the knives, then attach the knives to the strip, "
            "and finally hang the strip on the wall, so that the knives can be hung against the wall, "
            "which is convenient for storage, and can also prevent children from touching them, which is safer.",
            "A:The sun rises in the east because the earth rotates from west to east."]
        self.tokenizer = AutoTokenizer.from_pretrained(
            test_model_path,
            trust_remote_code=True,
            use_fast=False,
            model_max_length=self.model_max_length
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self._generate_test_partition_data()

    def test_preprocess(self):
        # Train mode
        sources_lens = _tokenize_fn(self.test_sources, self.tokenizer, mode='train')['input_ids_lens']
        result = preprocess(self.test_sources, self.test_targets, self.tokenizer, mode='train')
        for source_len, iids, tids in zip(sources_lens, result['input_ids'], result['labels']):
            self.assertEqual(iids.size(0), tids.size(0))
            self.assertLessEqual(iids.size(0), self.model_max_length)
            for ele in tids[:source_len]:
                self.assertEqual(ele, IGNORE_INDEX)
            if source_len < iids.size(0):
                self.assertNotEqual(tids[source_len], IGNORE_INDEX)
        # Other modes
        target_lens = _tokenize_fn(self.test_targets, self.tokenizer, mode="test")['input_ids_lens']
        result = preprocess(self.test_sources, self.test_targets, self.tokenizer, mode='test')
        for source_len, target_len, iids, tids in zip(sources_lens, target_lens, result['input_ids'],
                                                      result['labels']):
            self.assertEqual(iids.size(0), source_len)
            self.assertEqual(tids.size(0), target_len)

    def test_sft_data_manager(self):
        dm = LlaMaGenDataManger()
        with open(join(test_output_path, 'test_partition.pkl'), 'rb') as f:
            partition_obj = pickle.load(f)
        for cid in range(test_clients_num):
            train_dataset = dm.train_dataset_dict[cid]
            self.assertEqual(len(train_dataset), len(partition_obj['train'][cid]))
            cn = dm.coll_fn(None)
            dl = DataLoader(train_dataset, collate_fn=cn, batch_size=7)
            total_size = 0
            for idx, example in enumerate(dl):
                if idx != len(dl) - 1:
                    self.assertEqual(example['input_ids'].size(0), 7)
                total_size += example['input_ids'].size(0)
            self.assertEqual(total_size, len(train_dataset))

    def test_dop_data_manager(self):
        dm = DPODataManger()
        with open(join(test_output_path, 'test_partition.pkl'), 'rb') as f:
            partition_obj = pickle.load(f)
        for cid in range(test_clients_num):
            train_dataset = dm.train_dataset_dict[cid]
            self.assertEqual(len(train_dataset), len(partition_obj['train'][cid]))
            cn = dm.coll_fn(None)
            dl = DataLoader(train_dataset, collate_fn=cn, batch_size=7)
            total_size = 0
            for idx, example in enumerate(dl):
                if idx != len(dl) - 1:
                    self.assertEqual(example['chosen_input_ids'].size(0), 7)
                total_size += example['chosen_input_ids'].size(0)
            self.assertEqual(total_size, len(train_dataset))



if __name__ == '__main__':
    unittest.main(verbosity=2)
