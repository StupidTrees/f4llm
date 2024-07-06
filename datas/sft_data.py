import torch
import transformers

import copy
import random
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
from utils.register import registry
from datas.base_data import FedBaseDataManger

IGNORE_INDEX = -100
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "## Instruction:\n{instruction}\n## Input:\n{input}\n## Response:\n"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "## Instruction:\n{instruction}\n## Response:\n"
    ),
}
disc_template = """下面是一个司法法律相关的任务。输出一个适当地完成请求的响应。\n### 输入：{input}\n### 输出："""


def custom_pad_sequence(tensor_list, padding_value=-100, left_padding=True):
    # find the longest len
    max_length = max(len(t) for t in tensor_list)

    padded_list = []
    for tensor in tensor_list:
        padding_count = max_length - len(tensor)

        if left_padding:
            # left padding
            padded_tensor = torch.cat([torch.full((padding_count,), padding_value), tensor])
        else:
            # right padding
            padded_tensor = torch.cat([tensor, torch.full((padding_count,), padding_value)])
        padded_list.append(padded_tensor)

    padded_sequence = torch.stack(padded_list)

    return padded_sequence


def _tokenize_fn(strings, tokenizer, mode='train') -> Dict:
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


@registry.register_data("alpt")
class LlaMaGenDataManger(FedBaseDataManger):
    def __init__(self):
        super().__init__()
        self._load_data()

    def build_inputs(self, prompt_text, text):
        inputs_text = prompt_text.format(text)
        return inputs_text

    def process_examples(self, examples, mode="train", verbose=True):
        instances = []

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

                # input_ids = torch.nn.utils.rnn.pad_sequence(
                #     input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
                # )
                # labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
                return dict(
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
                )

        data_collator = DataCollatorForSupervisedDataset(tokenizer=self.tokenizer)

        return data_collator
