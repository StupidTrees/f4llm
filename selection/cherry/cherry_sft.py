import logging
import os
import sys
from dataclasses import dataclass
from typing import Dict, Sequence

import peft
import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer, BitsAndBytesConfig
sys.path.append(os.path.abspath('./'))
from tools.prompts.llama2_prompt import LLAMA_ALPACA_PROMPT_DICT
from datas.sft_data import preprocess
from selection.cherry.args import CherryModelArguments, CherryDataArguments, CherrySFTArguments, jload

IGNORE_INDEX = -100


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = jload(data_path)

        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = LLAMA_ALPACA_PROMPT_DICT["prompt_input"], LLAMA_ALPACA_PROMPT_DICT[
            "prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)
        logging.warning("Data loaded successfully")

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser((CherryModelArguments, CherryDataArguments, CherrySFTArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    bnb_config = None
    if training_args.quant < 32:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=training_args.quant == 8,  # load the model into memory using 8-bit precision
            load_in_4bit=training_args.quant == 4,  # load the model into memory using 4-bit precision
            bnb_4bit_use_double_quant=True,  # use double quantition
            bnb_4bit_quant_type="nf4",  # use NormalFloat quantition
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_8bit_use_double_quant=True,  # use double quantition
            bnb_8bit_quant_type="nf8",  # use NormalFloat quantition
            bnb_8bit_compute_dtype=torch.bfloat16  # use hf for computing when we need
        )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        device_map="auto",
        quantization_config=bnb_config,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    if training_args.use_lora:
        lora_config = peft.LoraConfig()
        model = peft.get_peft_model(model, lora_config)

    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    logging.warning("Training...")
    trainer.train()
    logging.warning("Saving model...")
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
