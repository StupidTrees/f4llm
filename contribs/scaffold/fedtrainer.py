
import copy
import torch
from transformers import TrainerCallback
from peft import get_peft_model_state_dict, set_peft_model_state_dict

from utils.register import registry
from utils.general import cosine_learning_rate
from trainers.FedBaseTrainer import BaseTrainer


def get_auxiliary_dict(fed_args, global_parameters):

    global_auxiliary = {}
    for key in global_parameters.keys():
        global_auxiliary[key] = torch.zeros_like(global_parameters[key])
    auxiliary_model_list = [copy.deepcopy(global_auxiliary) for _ in range(fed_args.clients_num)]
    auxiliary_delta_dict = [copy.deepcopy(global_auxiliary) for _ in range(fed_args.clients_num)]

    return global_auxiliary, auxiliary_model_list, auxiliary_delta_dict


class scaffold_callback(TrainerCallback):
    def __init__(self, correction, model):
        super(scaffold_callback, self).__init__()
        self.correction = correction
        self.model = model

    def on_step_end(self, args, state, control, **kwargs):
        model_para = copy.deepcopy(get_peft_model_state_dict(self.model))
        for name in model_para.keys():
            model_para[name] -= args.learning_rate * self.correction[name]
        set_peft_model_state_dict(self.model, model_para)


@registry.register_fedtrainer("scaffold")
class ScaffoldTrainer(BaseTrainer):
    def __init__(self, *args):
        super().__init__(*args)
        self._before_training()

        self.global_auxiliary, self.auxiliary_model_list, self.auxiliary_delta_dict = \
            get_auxiliary_dict(self.F, self.model_parameters)

        self.F.weight_type = "num"

    def _train_alone(self, idx, model_parameters, *args, **kwargs):
        self.logger.debug(f"\n{'=' * 35}\n>>> "
                          f"Client {idx} Trains in Round {self.round + 1}"
                          f" <<<\n{'=' * 35}")

        self.deserialize_model(model_parameters)
        train_dataset, eval_dataset = self.get_dataset(idx)

        # manually schedule the learning rate
        self.T.learning_rate = cosine_learning_rate(
            self.round, self.F.rounds, self.eval_args.learning_rate, 1e-6)

        if self.T.max_steps == -1:
            total_bs = registry.get("total_bs")
            max_steps = self.data.train_examples_num_dict[idx] // total_bs
            registry.register("max_steps", max_steps)

        # Initialize local Trainer
        train_op = registry.get_loctrainer(self.F.fl_algorithm)(
            model=self.model,
            args=self.T,
            train_dataset=train_dataset,
            tokenizer=self.data.tokenizer,
            data_collator=self.data.coll_fn(self.model),
            compute_metrics=self.metric.calculate_metric,
            global_parameters=model_parameters,
            local_auxiliary=self.auxiliary_model_list[idx],
            global_auxiliary=self.global_auxiliary
        )
        train_op.add_callback(scaffold_callback(train_op.correction, self.model))
        train_result = train_op.train()
        self.auxiliary_model_list[idx], self.auxiliary_delta_dict[idx] = train_op.get_auxiliary_param()
        del train_op

        train_loss = round(train_result.training_loss, 3)
        self.metric_log["train_logs"][self.round][idx] = train_loss
        self.logger.info(f">>> Client {idx} Train with lr {self.T.learning_rate*10000:.2f}e-4, Loss: {train_loss}")
        return train_loss

    def aggregator(self, serialized_params_list, weights=None):
        serialized_parameters = self.serialize_model_parameters()

        if weights is None:
            weights = [1.0 for _ in range(len(serialized_params_list))]

        total = sum(weights)
        weights = [weight/total for weight in weights]
        self.logger.info(f"This round clients' weights: {[round(weight, 3) for weight in weights]}")

        for key in serialized_parameters.keys():
            serialized_parameters[key] = sum(
                [client_params[key] * weights[client_id] for client_id, client_params in
                 enumerate(serialized_params_list)]
            )

        for key in self.global_auxiliary.keys():
            delta_auxiliary = sum([self.auxiliary_delta_dict[client_id][key] for client_id in self.client_ids])
            self.global_auxiliary[key] += delta_auxiliary / self.F.clients_num

        return serialized_parameters
