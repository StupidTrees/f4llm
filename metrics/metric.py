import re
import jieba
import pickle
import numpy as np
from rouge_chinese import Rouge
from utils.general import pickle_write
from utils.register import registry
from metrics.base_metric import BaseMetric
from rouge import Rouge
from transformers import AutoTokenizer
from bert_score import BERTScorer


@registry.register_metric("acc")
class AccForGenModel(BaseMetric):
    def __init__(self, tokenizer, is_decreased_valid_metric):
        super().__init__(tokenizer, is_decreased_valid_metric)

    def calculate_metric(self, eval_preds):
        preds, labels = eval_preds

        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        cnt = 0
        correct = 0
        for pred, label in zip(decoded_preds, decoded_labels):
            same_res = set(pred) & set(label)
            if len(same_res) == len(label):
                correct += 1
            cnt += 1

        score_dict = {"result": round(correct / cnt, 3)}
        return score_dict

    def update_metrics(self, *args):
        raise NotImplementedError

    @property
    def metric_name(self):
        return "Accuracy"


@registry.register_metric("lab")
class LABForGenModel(BaseMetric):
    def __init__(self, tokenizer, is_decreased_valid_metric):
        super().__init__(tokenizer, is_decreased_valid_metric)

    def calculate_metric(self, eval_preds):
        preds, labels = eval_preds

        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        score_dict = {
            "rouge1": [],
            "rouge2": [],
            "rougeL": []
        }
        for pred, label in zip(decoded_preds, decoded_labels):

            if len(pred) <= 0:
                continue

            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))
            rouge = Rouge()
            scores = rouge.get_scores(' '.join(hypothesis), ' '.join(reference))
            result = scores[0]

            # 司法摘要 score calculation, per https://github.com/china-ai-law-challenge/CAIL2022/tree/main/sfzy#%E8%AF%84
            # %E4%BB%B7%E6%96%B9%E5%BC%8F score_dict["averaged-rouge"].append(0.2*scores[0]['rouge-1'][
            # 'f']+0.3*scores[0]['rouge-2']['f']+0.5*scores[0]['rouge-l']['f'])
            score_dict["rouge1"].append(result['rouge-1']['f'])
            score_dict["rouge2"].append(result['rouge-2']['f'])
            score_dict["rougeL"].append(result['rouge-l']['f'])

        for k, v in score_dict.items():
            score_dict[k] = round(float(np.mean(v)), 3)

        score_dict = {"result": score_dict}
        return score_dict

    def update_metrics(self, *args):
        raise NotImplementedError

    @property
    def metric_name(self):
        return "rougeL"


@registry.register_metric("lcp")
class LCPForGenModel(BaseMetric):
    def __init__(self, tokenizer, is_decreased_valid_metric, save_outputs):
        super().__init__(tokenizer, is_decreased_valid_metric, save_outputs)

    def calculate_metric(self, eval_preds):
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
        save_data = {"eval_preds": eval_preds, "preds": decoded_preds,
                     "labels": decoded_labels, "inputs": decoded_inputs}
        pickle_write(save_data, checkpoint_opt_file)

        cnt = 0
        correct = 0
        all_cnts = {"right": 0, "true": 0, "pred": 0}
        for pred, label in zip(decoded_preds, decoded_labels):
            # em acc
            same_res = set(pred) & set(label)
            if len(same_res) == len(label):
                correct += 1
            cnt += 1

            # micro-f1
            all_cnts["right"] += len(same_res)
            all_cnts["pred"] += len(set(pred))
            all_cnts["true"] += len(set(label))

        recall = all_cnts["right"] / all_cnts["true"]
        precise = all_cnts["right"] / all_cnts["pred"] if all_cnts["pred"] else 0
        if precise + recall == 0:
            micro_f1 = 0
        else:
            micro_f1 = round(2 * recall * precise / (precise + recall)*100, 1)
        score_dict = {"result": {self.metric_name: round(correct / cnt*100, 1), "micro-f1": micro_f1}}
        return score_dict

    def update_metrics(self, *args):
        raise NotImplementedError

    @property
    def metric_name(self):
        return "accuracy"


@registry.register_metric("ler-f1")
class LERForGenModel(BaseMetric):
    def __init__(self, tokenizer, is_decreased_valid_metric):
        super().__init__(tokenizer, is_decreased_valid_metric)

    def format_output(self, preds):
        foutput = {}
        for pred in preds:
            if "：" in pred:
                info = pred.split("：")
                if len(info) == 2:
                    mention_type, mentions = info
                    mention_list = [mention for mention in mentions.split("，") if mention]
                    if len(mention_list) >= 1:
                        if mention_type not in foutput:
                            foutput[mention_type] = mention_list
                        else:
                            foutput[mention_type].extend(mention_list)
        return foutput

    def calculate_metric(self, eval_preds):
        preds, labels = eval_preds

        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # debug
        rounds = registry.get("round", 0)
        save_data = {"preds": decoded_preds, "labels": decoded_labels}
        with open(f"/userhome/output/test/ler/fedavg/{rounds}_eval.pkl", "wb") as file:
            pickle.dump(save_data, file)

        test_metric = dict()
        for preds, labels in zip(decoded_preds, decoded_labels):
            pred_list = [pred for pred in preds.split('\n') if pred]
            label_list = [label for label in labels.split('\n') if label]
            ## xxx:xxx,xxx\nxx:xx
            label_dicts = self.format_output(label_list)
            pred_dicts = self.format_output(pred_list)

            for mention_type in label_dicts:
                if mention_type not in test_metric:
                    test_metric[mention_type] = {"correct": 0, "pred": 0, "true": 0}
                true_mentions = label_dicts[mention_type]

                if getattr(pred_dicts, mention_type, None):
                    pred_mentions = pred_dicts[mention_type]
                    test_metric[mention_type]["true"] += len(true_mentions)
                    test_metric[mention_type]["pred"] += len(pred_mentions)
                    correct = 0
                    for pred_mention in pred_mentions:
                        if pred_mention in true_mentions:
                            correct += 1
                    test_metric[mention_type]["correct"] += correct
                else:
                    test_metric[mention_type]["true"] += len(true_mentions)
                    test_metric[mention_type]["pred"] += 0
                    test_metric[mention_type]["correct"] += 0

        # micro
        all_pred, all_true = 0, 0
        all_correct = 0
        for mention_type in test_metric:
            all_pred += test_metric[mention_type]["pred"]
            all_true += test_metric[mention_type]["true"]
            all_correct += test_metric[mention_type]["correct"]
        rec_ = all_correct / all_true if all_true != 0 else 0
        pre_ = all_correct / all_pred if all_pred != 0 else 0
        f1 = 2 * rec_ * pre_ / (rec_ + pre_) if rec_ != 0 or pre_ != 0 else 0
        results = {"result": f1}

        return results

    @property
    def metric_name(self):
        return "Micro-F1"


@registry.register_metric("medsi")
class MedSIForGenModel(BaseMetric):
    def __init__(self, tokenizer, is_decreased_valid_metric, save_outputs):
        super().__init__(tokenizer, is_decreased_valid_metric, save_outputs)

    def calculate_metric(self, eval_preds):
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
        return "bert-score"


@registry.register_metric("superni")
class SuperNI(BaseMetric):
    def __init__(self, tokenizer, is_decreased_valid_metric, save_outputs):
        super().__init__(tokenizer, is_decreased_valid_metric, save_outputs)
        self.scorer = Rouge()

    def calculate_metric(self, eval_preds):
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
        save_data = {"eval_preds": eval_preds, "preds": decoded_preds,
                     "labels": decoded_labels, "inputs": decoded_inputs}
        pickle_write(save_data, checkpoint_opt_file)

        mean_metrics = {"rouge-l": 0.0}
        rouge_1_scores, rouge_2_scores, rouge_l_scores = [], [], []
        for idx, decoded_pred in enumerate(decoded_preds):
            # if decoded_pred and decoded_labels[idx]:
            try:
                result = self.scorer.get_scores(decoded_pred, decoded_labels[idx])[0]
            except ValueError as e:
                # self.logger.critical(e)
                # self.logger.critical(f"pred: {decoded_pred}")
                # self.logger.critical(f"label: {decoded_labels[idx]}")
                rouge_1_scores.append(0.0)
                rouge_2_scores.append(0.0)
                rouge_l_scores.append(0.0)
                continue

            rouge_1_scores.append(round(result['rouge-1']['f'] * 100, 1))
            rouge_2_scores.append(round(result['rouge-2']['f'] * 100, 1))
            rouge_l_scores.append(round(result['rouge-l']['f'] * 100, 1))

        mean_metrics = {
            "rouge-1": round(np.mean(rouge_1_scores).item(), 1),
            "rouge-2": round(np.mean(rouge_2_scores).item(), 1),
            "rouge-l": round(np.mean(rouge_l_scores).item(), 1),
        }
        results = {"result": mean_metrics}
        return results

    def decoded(self, tensor):
        tensor = np.where(tensor != -100, tensor, self.tokenizer.pad_token_id)
        decoded_tensor = self.tokenizer.batch_decode(tensor, skip_special_tokens=True)
        return decoded_tensor

    @property
    def metric_name(self):
        return "rouge-l"


@registry.register_metric("p3")
class P3(BaseMetric):
    def __init__(self, tokenizer, is_decreased_valid_metric, save_outputs=True):
        super().__init__(tokenizer, is_decreased_valid_metric, save_outputs)
        self.all_tsk_name = [
            'anli', 'cb', 'copa',
            'rte', 'wic', 'winogrande', 'wsc'
        ]  # 'hellaswag'
        self.all_best_metrics = {}

    def calculate_metric(self, eval_preds):

        preds, labels, inputs = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = self.decoded(preds)
        decoded_labels = self.decoded(labels)
        decoded_inputs = self.decoded(inputs)

        all_tsk_output = {}
        for idx, decoded_pred in enumerate(decoded_preds):
            tsk_name = self.get_task_name(decoded_inputs[idx])
            dp = {
                "instruction": decoded_inputs[idx],
                "task": tsk_name,
                "pred": decoded_pred,
                "label": decoded_labels[idx],
            }
            if tsk_name not in all_tsk_output:
                all_tsk_output[tsk_name] = [dp]
            else:
                all_tsk_output[tsk_name].append(dp)

        checkpoint_opt_file = registry.get("checkpoint_opt_file")
        save_data = {"eval_preds": eval_preds, "preds": decoded_preds,
                     "labels": decoded_labels, "inputs": decoded_inputs,
                     "all_tsk_output": all_tsk_output}
        pickle_write(save_data, checkpoint_opt_file)

        all_tsk_metric = {}
        for tsk_name, tsk_dps in all_tsk_output.items():
            all_tsk_metric[tsk_name] = self.caclulate_acc(tsk_dps)
            if tsk_name not in self.all_best_metrics:
                self.all_best_metrics[tsk_name] = all_tsk_metric[tsk_name]
            else:
                if all_tsk_metric[tsk_name] > self.all_best_metrics[tsk_name]:
                    self.all_best_metrics[tsk_name] = all_tsk_metric[tsk_name]

        all_tsk_metric = dict(sorted(all_tsk_metric.items()))
        self.all_best_metrics = dict(sorted(self.all_best_metrics.items()))

        all_metrics = list(all_tsk_metric.values())
        all_tsk_metric[self.metric_name] = round(sum(all_metrics) / len(all_tsk_metric), 1)
        all_best_metrics = list(self.all_best_metrics.values())
        self.all_best_metrics[self.metric_name] = round(sum(all_best_metrics) / len(all_best_metrics), 1)
        self.logger.debug(f"all_tsk_metric: {all_tsk_metric}")
        self.logger.debug(f"best_metric: {self.all_best_metrics}")

        results = {"result": self.all_best_metrics}
        return results

    def get_task_name(self, input):
        pattern = r'\|\|(\w+)\|\|'
        tsk_name = re.findall(pattern, input)[0]
        return tsk_name

    def caclulate_acc(self, dps):
        right = 0
        for dp in dps:
            pred = dp["pred"]
            label = dp["label"]
            if pred == label:
                right += 1
        return round(right / len(dps) * 100, 1)

    def decoded(self, tensor):
        tensor = np.where(tensor != -100, tensor, self.tokenizer.pad_token_id)
        decoded_tensor = self.tokenizer.batch_decode(tensor, skip_special_tokens=True)
        return decoded_tensor

    @property
    def metric_name(self):
        return "accuracy"
