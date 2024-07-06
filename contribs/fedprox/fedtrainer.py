
from copy import deepcopy

from utils.register import registry
from utils.general import cosine_learning_rate
from trainers.FedBaseTrainer import BaseTrainer


@registry.register_fedtrainer("fedprox")
class FedProxTrainer(BaseTrainer):
    def __init__(self, *args):
        super().__init__(*args)
        self._before_training()

    def _train_alone(self, idx, model_parameters, *args, **kwargs):
        self.logger.debug(f"\n{'=' * 35}\n>>> Client {idx} Trains in "
                          f"Round {self.round + 1} <<<\n{'=' * 35}")

        self.deserialize_model(model_parameters)
        train_dataset, eval_dataset = self.get_dataset(idx)

        # manually schedule the learning rate
        self.T.learning_rate = cosine_learning_rate(
            self.round, self.F.rounds, self.eval_args.learning_rate, 1e-6)

        prox_mu = getattr(self.F, "prox_mu", 0.1)
        # Initialize local Trainer
        train_op = registry.get_loctrainer(self.F.fl_algorithm)(
            model=self.model,
            args=self.T,
            train_dataset=train_dataset,
            tokenizer=self.data.tokenizer,
            data_collator=self.data.coll_fn(self.model),
            compute_metrics=self.metric.calculate_metric,
            global_state=deepcopy(model_parameters),
            prox_mu=prox_mu
            # callbacks=None
            # optimizers
        )
        train_result = train_op.train()
        del train_op

        train_loss = round(train_result.training_loss, 3)
        self.metric_log["train_logs"][self.round][idx] = train_loss
        self.logger.info(f">>> Client {idx} Train with lr {self.T.learning_rate*10000:.2f}e-4, Loss: {train_loss}")
        return train_loss
