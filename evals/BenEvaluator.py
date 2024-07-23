
from utils.register import registry
from evals.BaseEvaluator import BaseEvaluator


@registry.register_eval("llm-eval")
class BenEvaluator(BaseEvaluator):
    def __init__(self, *args):
        super().__init__(*args)
        ...

    def on_eval(self):
        ...

    def on_eval_end(self):
        ...
