from abc import ABC

from utils.register import registry
from evals.BaseEvaluator import BaseEvaluator
from trainers.LocBaseSFT import LocalSFTTrainer
from trainers.LocBaseDPO import LocalDPOTrainer


@registry.register_eval("completion")
class SFTEvaluator(BaseEvaluator, ABC):
    def __init__(self, *args):
        super().__init__(*args)

    def build_eval_op(self, *args):
        eval_op = LocalSFTTrainer(
            model=self.model,
            args=self.eval_args,
            tokenizer=self.data.tokenizer,
            data_collator=self.data.coll_fn(self.model),
            compute_metrics=self.metric.calculate_metric,
        )
        return eval_op


@registry.register_eval("pairwise")
class DPOEvaluator(BaseEvaluator, ABC):
    def __init__(self, *args):
        super().__init__(*args)

    def build_eval_op(self, *args):
        eval_op = LocalDPOTrainer(
            model=self.model,
            args=self.eval_args,
            tokenizer=self.data.tokenizer,
            data_collator=self.data.coll_fn(self.model),
            compute_metrics=self.metric.calculate_metric,
        )
        return eval_op
