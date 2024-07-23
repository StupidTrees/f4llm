import torch
import transformers

import copy
import random
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
from utils.register import registry
from utils.general import custom_pad_sequence
from datas.base_data import FedBaseDataManger
from tools.prompts import all_prompts

IGNORE_INDEX = -100


def _tokenize_fn(strings, tokenizer, mode='train'):
    truncation = True if mode == "train" else False
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,  # usable when mode == train
            truncation=truncation,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
        sources,
        targets,
        tokenizer,
        mode="train"
):
    """Preprocess the data by tokenizing."""
    if mode == "train":
        examples = [s + t for s, t in zip(sources, targets)]
        examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
        input_ids = examples_tokenized["input_ids"]
        labels = copy.deepcopy(input_ids)
        for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
            label[:source_len] = IGNORE_INDEX
    else:
        sources_tokenized = _tokenize_fn(sources, tokenizer)
        targets_tokenized = _tokenize_fn(targets, tokenizer, mode)  # fix truncated label
        input_ids = sources_tokenized["input_ids"]
        labels = targets_tokenized["input_ids"]
    return dict(input_ids=input_ids, labels=labels)


@registry.register_data("llama_sft")
class LlaMaGenDataManger(FedBaseDataManger):
    def __init__(self):
        super().__init__()

    def build_inputs(self, prompt_text, text):
        inputs_text = prompt_text.format(text)
        return inputs_text

    def process_examples(self, examples, mode="train", verbose=True):
        instances = []
        PROMPT_DICT = all_prompts[self.data_config.template_name]
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]

        sources = [
            prompt_input.format_map(example) if example.get("input",
                                                            "<noinput>") != "<noinput>" else prompt_no_input.format_map(
                example)
            for example in examples
        ]
        # TODO response
        if mode == "train":
            target_key = 'response'
        # elif self.data_config.task_name in ["medsi"]:
        #     target_key = 'gpt-4-answer'
        elif self.data_config.task_name in ["superni", "medi", "alpt", "tde_alpt"]:
            target_key = 'response'
        else:
            target_key = 'output'

        targets = [f"{example[target_key]}{self.tokenizer.eos_token}" for example in examples]

        # if registry.get("round", 0) == 0:
        if mode == "train" and verbose:
            self.logger.info("=" * 40)
            self.logger.info(f"{mode} 0: {sources[0]} {targets[0]}")
            self.logger.info("=" * 40)

        data_dict = preprocess(sources, targets, self.tokenizer, mode)
        for idx, input_ids in enumerate(data_dict["input_ids"]):
            instances.append(
                {"input_ids": input_ids, "labels": data_dict["labels"][idx],
                 "idx": f"{mode}-{idx}", "example": examples[idx]}
            )
        return instances

    def coll_fn(self, model):
        # build data collection functions
        @dataclass
        class DataCollatorForSupervisedDataset(object):
            """Collate examples for supervised fine-tuning."""

            tokenizer: transformers.PreTrainedTokenizer

            def __call__(self, instances):
                left_padding = False if registry.get("phase") == "train" else True

                input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
                input_ids = custom_pad_sequence(input_ids, padding_value=self.tokenizer.pad_token_id,
                                                left_padding=left_padding)
                labels = custom_pad_sequence(labels, padding_value=self.tokenizer.pad_token_id,
                                             left_padding=left_padding)

                return dict(
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
                )

        data_collator = DataCollatorForSupervisedDataset(tokenizer=self.tokenizer)

        return data_collator
