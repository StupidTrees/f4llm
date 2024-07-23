"""Base DataLoader"""

import os
import numpy as np
from abc import ABC

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq

from utils.register import registry
from utils.general import pickle_read, pickle_write


class FedBaseDataset(Dataset):
    def __init__(self, features, **kv):
        self.data = features

        for k, v in kv.items():
            setattr(self, k, v)

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

    def select(self, max_samples):
        # max_samples = min(len(self), max_samples)
        features = self.data[0:max_samples]
        return FedBaseDataset(features)


class FedBaseDataManger(ABC):
    def __init__(self):

        config = registry.get("config")
        self.model_config = config.model_config
        self.data_config = config.data_config
        self.training_config = config.training_config
        self.federated_config = config.federated_config

        self.is_fl = config.is_fl
        self.partition_name = self.federated_config.partition_name
        self.clients_list = self.federated_config.clients_id_list
        self.logger = registry.get("logger")

        self.ignore_index = self.data_config.ignore_index
        self.model_max_length = self.data_config.model_max_length
        # self.max_tgt_len = self.training_config.generation_max_length

        self._load_attributes()
        self._build_tokenizer()
        self._build_registry()

    def load_data(self):

        train_dataset_dict, valid_dataset_dict, test_dataset_dict = {}, {}, {}
        train_features_all, valid_features_all, test_features_all = [], [], []

        train_features_dict, valid_features_dict, test_features_dict, train_examples_num_dict, \
        valid_examples_num_dict, test_examples_num_dict, train_num, valid_num, test_num = self._load_cached_data()

        for idx in range(self.attribute["clients_num"]):
            train_dataset_dict[idx] = self.build_dataset(train_features_dict[idx])
            valid_dataset_dict[idx] = self.build_dataset(valid_features_dict[idx]) \
                if len(valid_features_dict[idx]) != 0 else None
            test_dataset_dict[idx] = self.build_dataset(test_features_dict[idx]) \
                if len(test_features_dict[idx]) != 0 else None

            train_features_all += list(train_features_dict[idx])
            valid_features_all += list(valid_features_dict[idx])
            test_features_all += list(test_features_dict[idx])

        train_dataset_dict[-1] = self.build_dataset(train_features_all)
        valid_dataset_dict[-1] = self.build_dataset(valid_features_all) if valid_features_all else None
        test_dataset_dict[-1] = self.build_dataset(test_features_all) if test_dataset_dict else None

        self.train_dataset_dict = train_dataset_dict
        self.valid_dataset_dict = valid_dataset_dict
        self.test_dataset_dict = test_dataset_dict

        self.train_examples_num_dict = train_examples_num_dict
        self.logger.info(f"Train num: {self.train_num}, "
                         f"Valid num: {self.valid_num}, "
                         f"Test num: {self.test_num}")

    def _load_cached_data(self):
        with self.training_config.main_process_first(desc="Dataset pre-processing"):
            if not self.data_config.overwrite_cache and os.path.isfile(self.cached_data_file):
                self.logger.info(f"loading cached data from {self.cached_data_file}")
                train_features_dict, valid_features_dict, test_features_dict, train_examples_num_dict, \
                valid_examples_num_dict, test_examples_num_dict, self.train_num, self.valid_num, self.test_num \
                    = pickle_read(self.cached_data_file)
            else:
                self.logger.info(f"generating cached data ...")
                train_features_dict, valid_features_dict, test_features_dict, train_examples_num_dict, \
                valid_examples_num_dict, test_examples_num_dict, self.train_num, self.valid_num, self.test_num \
                    = self._convert_examples_to_features()

        return train_features_dict, valid_features_dict, test_features_dict, train_examples_num_dict, \
               valid_examples_num_dict, test_examples_num_dict, self.train_num, self.valid_num, self.test_num

    def _convert_examples_to_features(self):
        raw_data = pickle_read(self.data_config.raw_dataset_path)
        partition_data = pickle_read(self.data_config.partition_dataset_path)

        if self.partition_name in partition_data:
            partition_data = partition_data[self.partition_name]

        train_features_dict, valid_features_dict, test_features_dict = \
            {}, {}, {}
        train_examples_num_dict, valid_examples_num_dict, test_examples_num_dict = \
            {}, {}, {}

        n_clients = self.attribute["clients_num"]
        if n_clients != self.federated_config.clients_num:
            raise ValueError(f"partition data have {n_clients} clients "
                             f"that mismatches your input {self.federated_config.clients_num} clients")

        self.logger.info("convert train examples into features ...")
        train_features_all = np.array(self.process_examples(raw_data["train"], "train"))

        self.logger.info("convert valid examples into features ...")
        if "valid" not in raw_data:
            valid_features_all = []
        else:
            valid_features_all = np.array(self.process_examples(raw_data["valid"], "valid"))

        self.logger.info("convert test examples into features ...")
        if "test" not in raw_data:
            test_features_all = []
        else:
            test_features_all = np.array(self.process_examples(raw_data["test"], "test"))

        self.logger.info("build clients train & valid features ...")
        for idx in range(n_clients):
            client_train_list = partition_data["train"][idx]
            train_examples_num_dict[idx] = len(client_train_list)
            train_features_dict[idx] = train_features_all[client_train_list]

            if "valid" not in partition_data:
                client_valid_list = partition_data["valid"][idx]
                valid_examples_num_dict[idx] = len(client_valid_list)
                valid_features_dict[idx] = valid_features_all[client_valid_list]
            else:
                valid_examples_num_dict[idx], valid_features_dict[idx] = 0, []

            if "test" not in partition_data:
                client_test_list = partition_data["test"][idx]
                test_examples_num_dict[idx] = len(client_test_list)
                test_features_dict[idx] = test_features_all[client_test_list]
            else:
                test_examples_num_dict[idx], test_features_dict[idx] = 0, []

        self.train_num, self.valid_num, self.test_num = \
            len(train_features_all), len(valid_features_all), len(test_features_all)

        federated_data = (
            train_features_dict, valid_features_dict, test_features_dict,
            train_examples_num_dict, valid_examples_num_dict, test_examples_num_dict,
            self.train_num, self.valid_num, self.test_num
        )

        pickle_write(federated_data, self.cached_data_file)
        self.logger.info(f"processed features saved in {self.cached_data_file}")

        return federated_data

    def process_examples(self, examples, mode="train"):
        instances = []

        for idx, example in enumerate(examples):
            text, label = example["text_a"], example["label"]

            # build input with instructions
            input_text = self.build_inputs(self.prompt_text, text)
            src_ids = self.tokenizer.encode(input_text, add_special_tokens=False)
            # if len(src_ids) > self.data_config.max_src_length:
            #     src_ids = src_ids[:self.data_config.max_src_length]

            tgt_ids = self.tokenizer.encode(label, add_special_tokens=False)
            # if len(tgt_ids) > self.max_tgt_len:
            #     tgt_ids = tgt_ids[:self.max_tgt_len]

            context_length = len(src_ids)
            if mode == "train":
                input_ids = src_ids + tgt_ids + [self.tokenizer.eos_token_id]
                label_ids = [self.tokenizer.pad_token_id] * context_length + tgt_ids + [self.tokenizer.eos_token_id]
            else:
                input_ids = src_ids
                label_ids = tgt_ids

            input_ids = input_ids[: self.model_max_length]
            label_ids = label_ids[: self.model_max_length]

            # training/evaluate/predict with left padding -->  bad performance for baichuan2
            # pad_len = self.model_max_length - len(input_ids)
            # if mode == "train":
            #     input_ids = [self.tokenizer.pad_token_id] * pad_len + input_ids
            #     label_ids = [self.tokenizer.pad_token_id] * pad_len + label_ids
            # else:
            #     input_ids = [self.tokenizer.pad_token_id] * pad_len + input_ids
            # label_ids = [(l if l != self.tokenizer.pad_token_id else self.ignore_index) for l in label_ids]

            # training with right padding & evaluate/predict with left padding
            pad_len = self.model_max_length - len(input_ids)
            if mode == "train":
                input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
                label_ids = label_ids + [self.tokenizer.pad_token_id] * pad_len
            else:
                input_ids = [self.tokenizer.pad_token_id] * pad_len + input_ids
            label_ids = [(l if l != self.tokenizer.pad_token_id else self.ignore_index) for l in label_ids]

            input_ids = torch.LongTensor(input_ids)
            label_ids = torch.LongTensor(label_ids)
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()

            instances.append(
                {"input_ids": input_ids, "labels": label_ids,
                 "attention_mask": attention_mask, "idx": f"{mode}-{idx}"}
            )

        return instances

    def coll_fn(self, model):
        # build data collection functions
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=model,
            label_pad_token_id=self.ignore_index,
            pad_to_multiple_of=None,
            padding=False
        )
        return data_collator

    def _load_attributes(self):
        partition_data = pickle_read(self.data_config.partition_dataset_path)
        if self.partition_name in partition_data:
            partition_data = partition_data[self.partition_name]
        self.attribute = partition_data["attribute"]
        self.clients_num = self.attribute["clients_num"]

    def _build_registry(self):
        # used for xlms
        if 'lang_map' in self.attribute:
            registry.register("eval_batch", self.training_config.per_device_eval_batch_size)
            registry.register("lang_map", self.attribute["lang_map"])

    def build_dataset(self, features):
        dataset = FedBaseDataset(features)
        return dataset

    def _build_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.model_name_or_path,
            trust_remote_code=True,
            use_fast=False,
            model_max_length=self.model_max_length
        )
        if self.model_config.model_type in ["llama2-base", "tinyllama"]:
            self.tokenizer.pad_token = self.tokenizer.unk_token
        elif self.model_config.model_type in ["qwen"]:
            self.tokenizer.eos_token = self.tokenizer.decode(self.tokenizer.eod_id)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eod_id

    def build_inputs(self, prompt_text, text):
        inputs_text = prompt_text.format(text)
        return inputs_text

    @property
    def cached_data_file(self):
        cached_file_name = f"models={self.model_config.model_type}_" \
                           f"seq={self.data_config.model_max_length}_" \
                           f"clients={self.federated_config.clients_num}_" \
                           f"alpha={self.federated_config.alpha}"

        cached_file = os.path.join(
            self.data_config.cache_dir, cached_file_name
        )
        return cached_file

    @property
    def prompt_text(self):
        raise NotImplementedError
