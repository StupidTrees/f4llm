import numpy as np
import pandas as pd

from metrics.base_metric import BaseMetric
from utils.general import pickle_write
from utils.register import registry
from sklearn.metrics import accuracy_score


@registry.register_metric("openai")
class MetricForOpenAI(BaseMetric):
    def __init__(self, tokenizer, is_decreased_valid_metric, save_outputs):
        super().__init__(tokenizer, is_decreased_valid_metric, save_outputs)

    def calculate_metric(self, eval_preds):
        # save output for test openai
        if self.save_outputs:
            preds, labels, inputs = eval_preds
        else:
            preds, labels = eval_preds
            inputs = None

        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = self.decoded(preds)
        decoded_labels = self.decoded(labels)
        decoded_inputs = self.decoded(inputs) if self.save_outputs else None

        checkpoint_opt_file = registry.get("checkpoint_opt_file")
        save_data = {"preds": decoded_preds, "labels": decoded_labels, "inputs": decoded_inputs}
        pickle_write(save_data, checkpoint_opt_file)

        results = {"result": {self.metric_name: 0.0}}
        return results

    def decoded(self, tensor):
        tensor = np.where(tensor != -100, tensor, self.tokenizer.pad_token_id)
        decoded_tensor = self.tokenizer.batch_decode(tensor, skip_special_tokens=True)
        return decoded_tensor

    @property
    def metric_name(self):
        return "metric"


@registry.register_metric("pairwise")
class MetricForDPOPairwise(BaseMetric):
    def __init__(self, tokenizer, is_decreased_valid_metric, save_outputs):
        super().__init__(tokenizer, is_decreased_valid_metric, save_outputs)

    def calculate_metric(self, eval_preds):
        try:
            predictions = eval_preds.predictions
            preds = np.argmax(predictions, axis=1).reshape(-1)
        except:
            preds = eval_preds["preds"]

        labels = np.zeros(preds.shape)
        inputs = self.decoded(eval_preds.inputs) if self.save_outputs else None

        checkpoint_opt_file = registry.get("checkpoint_opt_file")
        save_data = {"preds": preds, "labels": labels, "inputs": inputs}
        pickle_write(save_data, checkpoint_opt_file)

        accuracy = float(accuracy_score(labels, preds, normalize=True))
        results = {"result": {self.metric_name: round(accuracy, 3)}}

        return results

    @property
    def metric_name(self):
        return "accuracy"


@registry.register_metric("rwben")
class MetricForRewardBenwise(BaseMetric):
    def __init__(self, tokenizer, is_decreased_valid_metric, save_outputs):
        super().__init__(tokenizer, is_decreased_valid_metric, save_outputs)

        self.set2idx = registry.get("set2idx")
        self.idx2set = registry.get("idx2set")

    def calculate_metric(self, eval_preds):
        try:
            predictions = eval_preds.predictions
            preds = np.argmax(predictions, axis=1).reshape(-1)
            set_ids = getattr(eval_preds, "label_ids")
        except:
            preds = eval_preds["preds"]
            set_ids = eval_preds["set_ids"]

        labels = np.zeros(preds.shape)
        inputs = self.decoded(eval_preds.inputs) if self.save_outputs else None

        checkpoint_opt_file = registry.get("checkpoint_opt_file")
        save_data = {"preds": preds, "labels": labels, "inputs": inputs, "set_ids": set_ids}
        pickle_write(save_data, checkpoint_opt_file)

        # print(type(set_ids))
        # set_ids = set_ids.numpy()

        # 1. 将输入的numpy arrays转换为pandas DataFrame
        df = pd.DataFrame({
            'labels': labels,
            'predicts': preds,
            'set_ids': set_ids
        })

        # 2. 使用groupby按照set_ids进行分组
        grouped = df.groupby('set_ids')

        # 3. 计算每个分组的准确率
        accuracy_per_set = grouped.apply(lambda x: (x['labels'] == x['predicts']).mean())
        accuracy_per_dict = accuracy_per_set.to_dict()

        bench_results = {self.idx2set[int(key)]: round(accuracy_per_dict[key], 3) for key in accuracy_per_dict}
        mean_acc = sum(list(accuracy_per_dict.values()))/len(list(accuracy_per_dict.values()))
        bench_results["mean accuracy"] = round(mean_acc, 3)
        results = {"result": bench_results}

        self.logger.debug(results)

        return results

    @property
    def metric_name(self):
        return "mean accuracy"
