import os
import copy
import time
import random
from abc import ABC
from glob import glob
from tabulate import tabulate
from copy import deepcopy
import pandas as pd

import torch
from peft import (
    get_peft_model_state_dict,
    set_peft_model_state_dict, load_peft_weights)

import grpc
from commus.message import Message
import commus.gRPC_comm_manager_pb2
import commus.gRPC_comm_manager_pb2_grpc
from commus.communicator import gRPCCommManager

from utils.register import registry
from utils.general import metric_save, is_best, pickle_read, run_process
from utils.general import setup_seed, is_petuning, cosine_learning_rate, LoadBalanceSampling
from utils.serialization import SerializationTool
from trainers.LocBaseTrainer import LocalBaseTrainer
from datas.base_data import FedBaseDataset

from contribs.centralized.miscs import CenEndEvalStepCallback, decoded_data


class BaseTrainer(ABC):
    def __init__(self, *args):

        config = registry.get("config")
        self.M = config.M
        self.D = config.D
        self.T = config.T
        self.F = config.F

        self.is_fl = config.is_fl
        self.client_num = len(config.F.clients_id_list)
        self.param_list = []  # used for global update
        self.loss_list = []  # used for loss-aware aggregate

        self.round = 0
        self.phase = registry.get("phase")
        self.logger = registry.get("logger")
        self.debug = registry.get("debug")

    def _build_data(self):
        self.data = registry.get_data_class(self.D.task_name)()

    def _build_model(self):
        self.model = registry.get_model_class(self.M.model_type)(
            task_name=self.D.task_name
        ).build_model()
        self.model_parameters = self.serialize_model_parameters()

    def _build_metric(self):
        self.logger.info(f"Metric name: {self.T.metric_name.upper()}, "
                         f"is_decreased_valid_metric: "
                         f"{self.T.is_decreased_valid_metric}")
        self.metric = registry.get_metric_class(self.T.metric_name)(
            self.data.tokenizer, self.T.is_decreased_valid_metric, self.T.save_outputs
        )
        self.metric_name = self.metric.metric_name

        # build federated eval-function' args
        self._build_eval_mode()

        # build metric_log_and_line
        self._build_general_metric_logs()

    def _build_general_metric_logs(self):
        self.metric_log = {
            "model_type": self.M.model_type,
            "tuning_type": self.M.tuning_type,
            "clients_num": self.F.clients_num,
            "alpha": self.F.alpha, "task": self.D.task_name,
            "fl_algorithm": self.F.fl_algorithm, "data": self.D.data_name,
            "info": registry.get("metric_line")[0:-1],
            "train_logs": [],
        }
        # metric line
        self.metric_line = registry.get("metric_line")

        self._build_metric_logs()

    def _build_metric_logs(self):
        self.metric_log["eval_logs"] = {}
        self.global_test_best_metric = 0.0
        self.global_valid_best_metric = \
            float("inf") if self.T.is_decreased_valid_metric else -float("inf")

    def _build_eval_mode(self):
        self.eval_args = copy.deepcopy(self.T)
        self.eval_args.evaluation_strategy = "epoch"
        self.eval_args.predict_with_generate = True

    def _build_selections(self):
        self.selections = []
        for i in range(self.F.rounds):
            self.selections.append(random.sample(
                range(self.F.client_num_in_total),
                self.F.client_num_per_round
            ))

    def _before_training(self):

        self.logger.info(f"{self.role} set seed {self.T.seed}")
        setup_seed(self.T.seed)

        self.logger.info(f"{self.role} building dataset ...")
        self._build_data()

        self.logger.info(f"{self.role} building model ...")
        self._build_model()

        # build metric
        self._build_metric()  # return computer metric

        # global model
        self.best_glo_params = self.serialize_model_parameters()

        # build client selection before hg trainer
        self._build_selections()

        # build communicators
        self._build_communicators()

    def _build_communicators(self):
        if self.role == "server":
            self.logger.debug(f"server build communicator")
            self.comm_manager = gRPCCommManager(
                host=self.F.server_ip,
                port=self.F.server_port,
                client_num=self.F.num_sub
            )
        else:
            time.sleep(2)  # wait for server
            self.logger.debug(f"subserver {self.F.client_name} build communicator")
            self.comm_manager = gRPCCommManager(
                host=self.F.client_ip,
                port=self.F.client_port,
                client_num=1,
                cfg=None
            )
            self.comm_manager.add_neighbors(
                neighbor_id=self.F.server_ip,
                address='{}:{}'.format(self.F.server_ip, self.F.server_port)
            )

    def _server_join(self):
        client_num = 0
        while client_num < self.F.num_sub:
            msg = self.comm_manager.receive()
            if msg.msg_type == "join_in":
                client_num += 1
                self.comm_manager.add_neighbors(neighbor_id=msg.sender,
                                                address=f"{msg.content['client_ip']}:{msg.content['client_port']}")
                self.logger.info(f"Subserver {msg.sender} joined in.")
                self.logger.info(list(self.comm_manager.neighbors.keys()))
        self.logger.debug("all subserver connect")

    def _client_join(self):
        self.comm_manager.send(
            Message(
                msg_type='join_in',
                sender=self.F.client_name,
                receiver=[self.F.server_ip],
                timestamp=0,
                content={
                    'client_ip': self.F.client_ip,
                    'client_port': self.F.client_port
                }
            )
        )

    def server_run(self):
        self._server_join()

        while self.round < self.F.rounds:
            self.client_selection()
            balance_sampling = LoadBalanceSampling(self.client_ids, self.F.num_sub)
            client_ids = {}
            for i in range(self.F.num_sub):
                client_ids[i] = balance_sampling[i]

            self.comm_manager.send(
                Message(
                    msg_type='update_param',
                    sender=0,
                    receiver=list(self.comm_manager.neighbors.keys()),
                    timestamp=0,
                    content={
                        'model': self.model_parameters,
                        'client_ids': client_ids
                    }
                )
            )

            num_sub = 0
            params_list, loss_list = [], []
            while num_sub < self.F.num_sub:
                msg = self.comm_manager.receive()
                if msg.msg_type == "update_param":
                    num_sub += 1
                    for client_id, params in msg.content['model'].items():
                        params_list.append(params)
                        loss_list.append(msg.content['loss'][client_id])
                        self.metric_log["train_logs"][self.round][client_id] = msg.content['loss'][client_id]

            # aggregation
            self.global_update(params_list, loss_list)

    def client_run(self):
        self._client_join()
        while True:
            msg = self.comm_manager.receive()
            if msg.msg_type == 'terminate':
                # self.test()
                break
            elif msg.msg_type == "update_param":
                model_parameters = msg['model']
                client_ids = msg['client_ids'][int(self.F.client_name)]
                self.local_update(client_ids, model_parameters)

    def run(self):
        self.logger.critical(f"{self.phase.upper()} START")
        if self.phase == "train":
            self.train()
        elif self.phase == "eval":
            self.eval()
        elif self.phase == "zst":
            self.zero_test()
        else:
            self.predict()

    def train(self):
        if self.role == "server":
            self.server_run()
            self.on_server_end()
        else:
            self.client_run()
            self.on_client_end()

    def cen_train(self, client_id=-1):
        train_dataset, eval_dataset = self.get_dataset(client_id)

        # set some parameters
        total_steps = len(train_dataset) / registry.get("total_bs") * self.T.num_train_epochs
        if self.F.log_valid_len:
            self.T.greater_is_better = False if self.T.is_decreased_valid_metric else True
            self.T.eval_steps = max(int(total_steps / self.F.log_valid_len), 1)

        if self.F.save_valid_len:
            self.T.save_strategy = "steps"
            self.T.save_steps = max(int(total_steps / self.T.num_train_epochs), 1)
            self.T.save_total_limit = int(self.T.num_train_epochs)
            self.T.evaluation_strategy = 'no'

        # Initialize Centralized Trainer
        train_op = registry.get_loctrainer("centralized")(
            model=self.model,
            args=self.T,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.data.tokenizer,
            data_collator=self.data.coll_fn(self.model),
            compute_metrics=self.metric.calculate_metric
        )
        train_op.add_callback(CenEndEvalStepCallback(
            train_op, metric_name=self.metric_name)
        )
        train_op.train()

    def fed_train(self):

        # while self.round < self.F.rounds:
        #     self.client_selection()
        #     self.local_update()
        #     self.global_update()
        ...

    def client_selection(self):
        self.client_ids = self.selections[self.round]
        self.metric_log["train_logs"].append([0.0 for _ in range(self.F.client_num_in_total)])
        self.logger.critical(f"Round {self.round + 1} start, Selected Clients: {self.client_ids}")

    def local_update(self, client_ids, model_parameters):
        param_list, loss_list = {}, {}
        for idx in client_ids:
            train_loss = self._train_alone(
                idx=idx,
                model_parameters=model_parameters
            )
            updated_model_parameters = self.serialize_model_parameters()

            if self.F.use_ldp:
                for key in updated_model_parameters:
                    updated_model_parameters[key] += \
                        torch.randn(updated_model_parameters[key].shape).to("cuda") * self.F.ldp_delta
                    # TODO DP-Noisy Normalized

            param_list[idx] = updated_model_parameters
            loss_list[idx] = train_loss

        self.comm_manager.send(
            Message(
                msg_type='update_param',
                sender=self.F.client_name,
                receiver=[self.F.server_ip],
                timestamp=0,
                content={
                    'model': param_list,
                    'loss': loss_list
                }
            )
        )

    def _train_alone(self, idx, model_parameters, *args, **kwargs):
        self.logger.debug(f"\n{'=' * 35}\n>>> Subserver {self.F.client_name} with "
                          f"Client {idx} Trains in Round {self.round + 1} <<<\n{'=' * 35}")

        self.deserialize_model(model_parameters)
        train_dataset, eval_dataset = self.get_dataset(idx)

        # manually schedule the learning rate
        self.T.learning_rate = cosine_learning_rate(
            self.round, self.F.rounds, self.eval_args.learning_rate, 1e-6)

        # Initialize local Trainer
        train_op = registry.get_loctrainer(self.F.local_trainer_name)(
            model=self.model,
            args=self.T,
            train_dataset=train_dataset,
            tokenizer=self.data.tokenizer,
            data_collator=self.data.coll_fn(self.model),
            compute_metrics=self.metric.calculate_metric,
            # callbacks=None
            # optimizers
        )
        train_result = train_op.train()
        del train_op

        train_loss = round(train_result.training_loss, 3)
        self.logger.info(f">>> Subserver {self.F.client_name} Client {idx} Train with lr "
                         f"{self.T.learning_rate*10000:.2f}e-4, Loss: {train_loss}")
        return train_loss

    def global_update(self, param_list, loss_list):
        assert len(param_list) <= self.F.client_num_per_round

        self.round += 1
        should_eval, should_save = False, False
        if self.F.log_valid_len and self.round % self.F.log_valid_len == 0:
            should_eval = True
        if self.F.save_valid_len and self.round % self.F.save_valid_len == 0:
            should_save = True

        this_round_loss = sum(loss_list)/len(loss_list)
        self.logger.warning(
            f"{self.F.fl_algorithm}-Round {self.round} with {len(param_list)} client updates aggregation, "
            f"This Round Evals: {should_eval}, This Round Save: {should_save}, This Round Loss: {this_round_loss:.3f}"
        )

        # Global Aggregation
        if self.F.weight_type == "num":
            weights = [self.data.train_examples_num_dict[client_id] for client_id in self.client_ids]
        else:
            weights = None
        serialized_parameters = self.aggregator(param_list, weights)
        self.deserialize_model(serialized_parameters)

        if should_eval:
            self.fed_valid()
            # test
            # self.on_train_end()
            # self.deserialize_model(serialized_parameters)

        if should_save:
            self.model_save(serialized_parameters)

        registry.register("round", self.round)
        self.model_parameters = copy.deepcopy(serialized_parameters)

    def fed_valid(self, idx=-1):
        self.logger.info(f"Round {self.round} Valid Start")
        _, eval_dataset = self.get_dataset(idx)

        eval_result = self.eval_fun(eval_dataset)
        self.on_round_end(eval_result)
        return eval_result

    def eval_fun(self, eval_dataset, checkpoint_file=None):
        # TODO: peft
        if checkpoint_file is not None:
            ckt_param = load_peft_weights(checkpoint_file)
            self.deserialize_model(ckt_param)

        # Initialize Eval Trainer
        eval_op = LocalBaseTrainer(
            model=self.model,
            args=self.eval_args,
            tokenizer=self.data.tokenizer,
            data_collator=self.data.coll_fn(self.model),
            compute_metrics=self.metric.calculate_metric,
        )
        eval_result = eval_op.evaluate(
            eval_dataset,
        )
        del eval_op
        return eval_result

    def zero_test(self):

        self.deserialize_model(self.best_glo_params)
        _, test_dataset = self.get_dataset(-1, "test")

        test_metric = self.eval_fun(
            eval_dataset=test_dataset)["eval_result"]

        global_test_best_metric = ""
        for metric_name, metric in test_metric.items():
            global_test_best_metric += f"{metric_name}={metric:.3f}_"
        self.global_test_best_metric = global_test_best_metric[0:-1]

        self.logger.debug(f"zst metric: {self.global_test_best_metric}")
        self.metric_save()

    def get_dataset(self, client_id, mode="train"):
        """Get :class:`Dataset` for ``client_id``."""
        train_dataset = self.data.train_dataset_dict[client_id]

        if mode == "train":
            valid_dataset_dict = self.data.valid_dataset_dict
        else:
            valid_dataset_dict = self.data.test_dataset_dict

        if client_id == -1 or self.F.pson:
            eval_dataset = valid_dataset_dict[client_id]
        else:
            eval_dataset = None

        if self.debug:
            if isinstance(train_dataset, FedBaseDataset):
                max_train_samples = min(len(train_dataset), self.D.max_train_samples)
            else:
                max_train_samples = range(min(len(train_dataset), self.D.max_train_samples))

            train_dataset = train_dataset.select(max_train_samples)
            if eval_dataset is not None:
                max_eval_samples = min(len(eval_dataset), self.D.max_eval_samples)
                eval_dataset = eval_dataset.select(max_eval_samples)
            else:
                max_eval_samples = 0
            self.logger.warning(f"Debug Mode Enable, Train Num: {max_train_samples}, "
                                f"Eval Num: {max_eval_samples}")

        return train_dataset, eval_dataset

    def serialize_model_parameters(self):
        if is_petuning(self.M.tuning_type):
            # model_parameters = SerializationTool.serialize_peft_model(
            #     self.model, tuning_type=self.M.tuning_type)
            model_parameters = deepcopy(get_peft_model_state_dict(self.model))
        else:
            model_parameters = SerializationTool.serialize_model(self.model)
        return model_parameters

    def deserialize_model(self, serialized_parameters):
        if is_petuning(self.M.tuning_type):
            set_peft_model_state_dict(self.model, serialized_parameters)
        else:
            SerializationTool.deserialize_model(self.model, serialized_parameters)

    # fl algorithm default fedavg
    def aggregator(self, serialized_params_list, weights=None):
        serialized_parameters = self.serialize_model_parameters()

        if weights is None:
            weights = [1.0 for _ in range(len(serialized_params_list))]

        total = sum(weights)
        weights = [weight/total for weight in weights]
        self.logger.info(f"This round clients' weights: {[round(weight, 3) for weight in weights]}")

        for key in serialized_parameters.keys():
            serialized_parameters[key] = sum(
                [serialized_params_list[client][key] * weights[client] for client in
                 range(len(serialized_params_list))])

        return serialized_parameters

    # end function
    def on_round_end(self, result):
        """Used for Global Update Logging"""
        test_metrics = result["eval_result"]
        test_metric = test_metrics[self.metric_name]

        with self.T.main_process_first(desc="Model & Metric Saving"):
            if is_best(self.global_valid_best_metric, test_metric, self.metric.is_decreased_valid_metric):
                self.global_valid_best_metric = test_metric
                self.best_glo_params = self.serialize_model_parameters()

        # base train info
        self.logger.info(f"{self.D.task_name}-{self.M.model_type} "
                         f"train with client={self.F.clients_num}_"
                         f"alpha={self.F.alpha}_"
                         f"epoch={self.T.num_train_epochs}_"
                         f"seed={self.T.seed}_"
                         f"comm_round={self.F.rounds}")

        self.logger.debug(f"{self.F.fl_algorithm} Eval Round:{self.round}, "
                          f"Current {self.metric_name}:{test_metric:.3f}, "
                          f"Best {self.metric_name}:{self.global_valid_best_metric:.3f}")

        self.metric_log["eval_logs"][f"round_{self.round}"] = test_metrics

    def on_eval_end(self, result):
        ...

    def on_server_end(self):
        """Using best parameters for prediction"""
        self.global_test_best_metric = ""
        if not self.is_fl or self.F.save_valid_len:
            self.best_glo_params = registry.get("best_glo_params", self.model_parameters)
            self.global_valid_best_metric = registry.get("best_valid_metric", 0.0)

        if self.F.save_valid_len:
            if not self.debug:
                metric_save(self, self.T, self.logger)
            self.logger.critical(f"Train done, Please Eval and Test in {self.T.checkpoint_dir}")
            return

        self.logger.critical("Final Test Start")
        self.deserialize_model(self.best_glo_params)
        _, test_dataset = self.get_dataset(-1, "test")
        test_result = self.eval_fun(test_dataset)
        global_test_best_metric = test_result["eval_result"]

        for metric_name, metric in global_test_best_metric.items():
            self.global_test_best_metric += f"{metric_name}={metric:.3f}_"
        self.global_test_best_metric = self.global_test_best_metric[0:-1]

        self.logger.critical(f"{self.F.fl_algorithm.upper()} Test, "
                             f"Best Model Metric, {self.global_test_best_metric}")
        self.save_all()

        self.comm_manager.send(
            Message(
                msg_type='terminate',
                sender=0,
                receiver=list(self.comm_manager.neighbors.keys()),
                timestamp=0,
                content={
                    '': '',
                }
            )
        )

    def on_client_end(self):
        self.logger.critical(f"Subserver {self.F.client_name} Train done")

    def save_all(self):
        self.metric_save()
        self.model_save()

    def metric_save(self):
        if self.debug:
            return
        metric_save(self, self.T, self.logger)

    def model_save(self, serialized_parameters=None):
        if self.phase != "train" or self.debug:
            return

        if self.F.save_valid_len:
            checkpoint_file = os.path.join(self.T.checkpoint_dir, f"round-{self.round}")
            self.deserialize_model(serialized_parameters)
            save_op = LocalBaseTrainer(
                model=self.model,
                args=self.eval_args,
            )
            save_op.save_model(checkpoint_file)
            self.logger.debug(f"Model Saved in: {checkpoint_file}")
        else:
            torch.save(self.best_glo_params, self.T.checkpoint_file)

    def on_test_end(self):
        ...

    def eval(self):

        best_file = None
        single_file = False

        pattern_name = "round*" if self.is_fl else "steps*"
        pattern = os.path.join(self.T.checkpoint_file, pattern_name)
        checkpoint_files = sorted(glob(pattern, recursive=True),
                                  key=lambda x: os.path.getctime(x), reverse=False)

        if len(checkpoint_files) == 0:
            # single checkpoint test
            self.logger.debug("Eval Single Checkpoint")
            checkpoint_files = [self.T.checkpoint_file]
            single_file = True

        ckpt_metric = {}
        for checkpoint_file in checkpoint_files:
            file = checkpoint_file.split("/")[-1]
            self.logger.info(f"Eval {file} Start")

            eval_key = registry.get("eval_key", "train")
            _, valid_dataset = self.get_dataset(-1, eval_key)

            # if self.T.save_outputs:
            checkpoint_opt_file = os.path.join(checkpoint_file, f"{file}.result.pkl")
            registry.register('checkpoint_opt_file', checkpoint_opt_file)

            if self.T.eval_reuse and os.path.exists(checkpoint_opt_file):
                eval_preds = pickle_read(checkpoint_opt_file)["eval_preds"]
                valid_metric = self.metric.calculate_metric(eval_preds)["result"]
            else:
                valid_metric = self.eval_fun(
                    eval_dataset=valid_dataset, checkpoint_file=checkpoint_file)["eval_result"]

            metric_value = valid_metric[self.metric_name]
            if is_best(self.global_valid_best_metric, metric_value, self.metric.is_decreased_valid_metric):
                self.global_valid_best_metric = metric_value
                self.best_glo_params = self.serialize_model_parameters()
                best_file = file

            ckpt_metric[file] = {self.metric_name: metric_value}
            self.logger.info(f"Model: {file}, Metric: {metric_value:.3f}, "
                             f"Best Model: {best_file}, "
                             f"Best: {self.global_valid_best_metric:.3f}")
            self.logger.info(f"Eval Results save in {checkpoint_opt_file}")

        if not single_file:
            metric_path = os.path.join(self.T.checkpoint_file, "metric.csv")
            metrics_df = pd.DataFrame.from_dict(ckpt_metric, orient='index')
            metrics_df["mean"] = metrics_df.mean(axis=1).round(1)
            sorted_metrics_df = metrics_df.sort_values(by='mean', ascending=False)
            sorted_metrics_df.reset_index(inplace=True)
            sorted_metrics_df.columns = ["round"] + list(sorted_metrics_df.columns[1:])

            sorted_metrics_df.to_csv(metric_path, sep="\t", index=False)
            self.logger.info(f"Metric outputs saved to {metric_path}")

            self.logger.info(f"\n{tabulate(sorted_metrics_df, headers='keys', tablefmt='pretty', showindex=False)}")
            self.logger.info(f"Copy Results")
            with self.T.main_process_first():
                run_process(f"cat {metric_path}")

        # self.deserialize_model(self.best_glo_params)
        # best/selected for test prediction
        # self.logger.critical("Test Start")
        # _, test_dataset = self.get_dataset(-1, "test")
        # test_result = self.eval_fun(test_dataset)
        # global_test_best_metric = test_result["eval_result"]
        #
        # self.global_test_best_metric = ""
        # for metric_name, metric in global_test_best_metric.items():
        #     self.global_test_best_metric += f"{metric_name}={metric:.3f}_"
        # self.global_test_best_metric = self.global_test_best_metric[0:-1]
        #
        # self.logger.critical(f"Test Done, "
        #                      f"Checkpoint Metric: {self.global_test_best_metric}, "
        #                      f"Model Path: {self.T.checkpoint_file}")
        #
        # if os.path.isdir(self.T.checkpoint_file):
        #     self.metric_save()

    def predict(self):
        ...

    @property
    def role(self):
        return self.F.role
