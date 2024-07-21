import torch
import transformers
from datasets import Dataset
from dataclasses import dataclass

from utils.register import registry
from utils.general import custom_pad_sequence
from datas.base_data import FedBaseDataManger, FedBaseDataset
from tools.prompts import all_prompts

IGNORE_INDEX = -100


@registry.register_data("dpo")
class DPODataManger(FedBaseDataManger):
    def __init__(self):
        super().__init__()
        self._load_data()

    def build_inputs(self, prompt_text, text):
        inputs_text = prompt_text.format(text)
        return inputs_text

    def process_examples(self, examples, mode="train", verbose=True):
        instances = []
        columns = list(examples[0].keys())
        template = all_prompts[self.data_config.template_name]

        max_model_length = self.data_config.model_max_length
        max_prompt_length = self.data_config.max_prompt_length
        max_response_length = max_model_length - max_prompt_length

        for idx, example in enumerate(examples):
            if 'chosen' not in columns or 'rejected' not in columns:
                assert 'instruction' in columns and 'input' in columns and 'output' in columns
                instruction, input, output = example['instruction'], example['input'], example['output']
                if input is not None and input != "":
                    instruction = instruction + '\n' + input
                assert len(output) > 1
                prompt, chosen, rejected = instruction, output[0], output[1]
            else:
                assert 'prompt' in columns and 'rejected' in columns and 'chosen' in columns
                prompt, chosen, rejected = example['prompt'], example['chosen'], example['rejected']

            source = template.format_map({'Instruction': prompt})
            source_ids = self.tokenizer.encode(text=source, add_special_tokens=False)
            chosen_ids = self.tokenizer.encode(text=chosen, add_special_tokens=False)
            rejected_ids = self.tokenizer.encode(text=rejected, add_special_tokens=False)

            if len(source_ids) > max_prompt_length - 1:
                source_ids = source_ids[:max_prompt_length - 1]
            if len(chosen_ids) > max_response_length - 1:
                chosen_ids = chosen_ids[:max_response_length - 1]
            if len(rejected_ids) > max_response_length - 1:
                rejected_ids = rejected_ids[:max_response_length - 1]

            source_chosen_ids = source_ids + [self.tokenizer.bos_token_id] + chosen_ids + [
                self.tokenizer.eos_token_id]
            source_chosen_labels = [IGNORE_INDEX] * len(source_ids) + [self.tokenizer.bos_token_id] + chosen_ids + [
                self.tokenizer.eos_token_id]
            source_rejected_ids = source_ids + [self.tokenizer.bos_token_id] + rejected_ids + [
                self.tokenizer.eos_token_id]
            source_rejected_labels = [IGNORE_INDEX] * len(source_ids) + [self.tokenizer.bos_token_id] + rejected_ids + [
                self.tokenizer.eos_token_id]

            source_chosen_length, source_rejected_length = len(source_chosen_ids), len(source_rejected_ids)
            max_length = max(source_chosen_length, source_rejected_length)

            source_chosen_ids = source_chosen_ids + [self.tokenizer.pad_token_id] * (
                    max_length - source_chosen_length)
            source_chosen_labels = source_chosen_labels + [IGNORE_INDEX] * (max_length - source_chosen_length)
            source_rejected_ids = source_rejected_ids + [self.tokenizer.pad_token_id] * (
                    max_length - source_rejected_length)
            source_rejected_labels = source_rejected_labels + [IGNORE_INDEX] * (max_length - source_rejected_length)

            inputs_ids = source_chosen_ids + source_rejected_ids
            labels = source_chosen_labels + source_rejected_labels
            instances.append({"idx": f"{mode}-{idx}",
                              "input_ids": inputs_ids, "labels": labels
                              })
        return instances

    def coll_fn(self, model):
        # pairwise data collator
        @dataclass
        class DataCollatorForPairwiseDataset(object):
            """Collate examples for pairwise dataset."""

            tokenizer: transformers.PreTrainedTokenizer

            def __call__(self, instances):
                chosen_ids, chosen_labels, rejected_ids, rejected_labels = [], [], [], []
                for instance in instances:
                    length = len(instance["input_ids"]) // 2
                    chosen_id = instance["input_ids"][:length]
                    rejected_id = instance["input_ids"][length:]
                    chosen_label = instance["labels"][:length]
                    rejected_label = instance["labels"][length:]

                    chosen_ids.append(torch.LongTensor(chosen_id))
                    chosen_labels.append(torch.LongTensor(chosen_label))
                    rejected_ids.append(torch.LongTensor(rejected_id))
                    rejected_labels.append(torch.LongTensor(rejected_label))

                chosen_input_ids = custom_pad_sequence(chosen_ids, padding_value=self.tokenizer.pad_token_id,
                                                       left_padding=True)
                chosen_labels = custom_pad_sequence(chosen_labels, padding_value=self.tokenizer.pad_token_id,
                                                    left_padding=True)
                rejected_input_ids = custom_pad_sequence(rejected_ids, padding_value=self.tokenizer.pad_token_id,
                                                         left_padding=True)
                rejected_labels = custom_pad_sequence(rejected_labels, padding_value=self.tokenizer.pad_token_id,
                                                      left_padding=True)

                return dict(
                    chosen_input_ids=chosen_input_ids,
                    chosen_labels=chosen_labels,
                    chosen_attention_mask=chosen_input_ids.ne(self.tokenizer.pad_token_id),
                    rejected_input_ids=rejected_input_ids,
                    rejected_labels=rejected_labels,
                    rejected_attention_mask=rejected_input_ids.ne(self.tokenizer.pad_token_id),
                )

        data_collator = DataCollatorForPairwiseDataset(tokenizer=self.tokenizer)

        return data_collator


@registry.register_data("ultrafeedback_binarized")
class UFBDataManger(FedBaseDataManger):
    def __init__(self):
        super().__init__()
        self._load_data()

    def build_inputs(self, prompt_text, text):
        inputs_text = prompt_text.format(text)
        return inputs_text

    def process_examples(self, examples, mode="train", verbose=True):
        instances = []
        for idx, example in enumerate(examples):
            instances.append(
                {"idx": f"{mode}-{idx}", "example": example}
            )
        return instances

    def build_dataset(self, features):
        data = {"prompt": [], "chosen": [], "rejected": []}
        for feature in features:
            example = feature["example"]
            for key in example:
                data[key].append(example[key])
        dataset = Dataset.from_dict(data)
        return dataset

    def coll_fn(self, model):
        return None
