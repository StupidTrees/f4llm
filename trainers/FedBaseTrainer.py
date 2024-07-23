import os
import copy
import time
import random
import torch

from commus.message import Message
from commus.communicator import gRPCCommunicationManager
from contribs.centralized.miscs import CenEndEvalStepCallback

from utils.register import registry
from utils.general import metric_save, is_best, pickle_read, run_process
from utils.general import setup_seed, cosine_learning_rate, LoadBalanceSampling

from trainers.BaseEngine import BaseEngine
from trainers.LocBaseSFT import LocalSFTTrainer


class BaseTrainer(BaseEngine):
    def __init__(self, *args):
        super().__init__(*args)

    def _build_selections(self):
        # TODO 添加激励未参加用户的采样方式
        self.selections = []
        for i in range(self.F.rounds):
            self.selections.append(random.sample(
                range(self.F.client_num_in_total),
                self.F.client_num_per_round
            ))

    def _build_communicators(self):
        if self.is_fl:
            self.logger.info(f"{self.role} building communicators ...")
            if self.role == "server":
                self.logger.debug(f"server build communicator")
                self.comm_manager = gRPCCommunicationManager(
                    ip=self.F.server_ip,
                    port=self.F.server_port,
                    max_connection_num=self.F.num_sub
                )
            else:
                time.sleep(5)  # wait for server
                self.logger.debug(f"subserver {self.F.client_name} build communicator")
                self.comm_manager = gRPCCommunicationManager(
                    ip=self.F.client_ip,
                    port=self.F.client_port,
                    max_connection_num=1,
                )
                self.comm_manager.add_communicator(
                    communicator_id=self.F.server_ip,
                    communicator_address='{}:{}'.format(self.F.server_ip, self.F.server_port)
                )
        else:
            self.logger.info("local or central training")

    def _before_training(self):
        # set seed
        setup_seed(self.T.seed)

        # build dataset and dataloader
        self._build_data()

        # build federated model
        self._build_model()

        # build metric
        self._build_metric()  # return computer metric

        # global model
        self.best_glo_params = self.serialize_model_parameters()

        # build client selection before building loc trainer
        self._build_selections()

        # build communicators
        self._build_communicators()

    def run(self):
        self.logger.critical(f" {self.role.upper()} {self.phase.upper()} START")
        if self.is_fl:
            self.server_run() if self.role == "server" else self.client_run()
        else:
            self.cen_train()

    def server_join(self):
        client_num = 0
        while client_num < self.F.num_sub:
            msg = self.comm_manager.receive()
            if msg.message_type == 100:
                client_num += 1
                self.comm_manager.add_communicator(
                    communicator_id=msg.sender,
                    communicator_address=f"{msg.content['client_ip']}:{msg.content['client_port']}")
                self.logger.info(f"Subserver {msg.sender} joined in.")
                self.logger.info(list(self.comm_manager.communicators.keys()))
        self.logger.debug("all subserver connect")

    def server_run(self):
        self.server_join()

        while self.round < self.F.rounds:
            # TODO server select client
            self.client_ids = self.selections[self.round]
            self.metric_log["train_logs"].append([0.0 for _ in range(self.F.client_num_in_total)])
            self.logger.critical(f"Round {self.round + 1} start, Selected Clients: {self.client_ids}")
            balance_sampling = LoadBalanceSampling(self.client_ids, self.F.num_sub)
            client_ids = {}
            for i in range(self.F.num_sub):
                client_ids[i] = balance_sampling[i]

            self.comm_manager.send(
                Message(
                    message_type=200,
                    sender="0",
                    receiver=list(self.comm_manager.communicators.keys()),
                    content={
                        'model': self.model_parameters,
                        'client_ids': client_ids,
                        'round': self.round
                    }
                )
            )

            num_sub = 0
            params_list, loss_list = [], []
            while num_sub < self.F.num_sub:
                msg = self.comm_manager.receive()
                if msg.message_type == 200:
                    num_sub += 1
                    for client_id, params in msg.content['model'].items():
                        params_list.append(params)
                        loss_list.append(msg.content['loss'][client_id])
                        self.metric_log["train_logs"][self.round][client_id] = msg.content['loss'][client_id]

            # aggregation
            self.global_update(params_list, loss_list)

        self.on_server_end()

    def client_join(self):
        self.comm_manager.send(
            Message(
                message_type=100,
                sender=self.F.client_name,
                receiver=[self.F.server_ip],
                content={
                    'client_ip': self.F.client_ip,
                    'client_port': self.F.client_port
                }
            )
        )

    def client_run(self):
        # client join in federated learning
        self.client_join()

        while True:
            msg = self.comm_manager.receive()
            if msg.message_type == 101:
                # quit federated learning
                self.on_client_end()
                break
            elif msg.message_type == 200:
                model_parameters = msg.content['model']
                self.round = msg.content['round']
                # TODO 直接传递list
                client_ids = msg.content['client_ids'][int(self.F.client_name)]
                self.local_process(client_ids, model_parameters)

    def cen_train(self, client_id=-1):

        # get local train and eval dataset
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

    def local_process(self, client_ids, model_parameters):
        param_list, loss_list = {}, {}
        for idx in client_ids:
            train_loss = self.local_train(
                idx=idx,
                model_parameters=model_parameters
            )
            updated_model_parameters = self.serialize_model_parameters()
            param_list[idx] = updated_model_parameters
            loss_list[idx] = train_loss

        self.comm_manager.send(
            Message(
                message_type=200,
                sender=self.F.client_name,
                receiver=[self.F.server_ip],
                content={
                    'model': param_list,
                    'loss': loss_list
                }
            )
        )

    def local_train(self, idx, model_parameters, *args, **kwargs):
        self.logger.debug(f"\n{'=' * 37}\n>>> Subserver={self.F.client_name}_"
                          f"Client={idx}_Round={self.round + 1} <<<\n{'=' * 37}")

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
        self.logger.info(f">>> Subserver={self.F.client_name}_Client={idx}_lr="
                         f"{self.T.learning_rate*10000:.2f}e-4_Loss={train_loss}")
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
            f"FL={self.F.fl_algorithm}_Round={self.round}_ClientNum={len(param_list)}_"
            f"Evals={should_eval}_Save={should_save}_Loss={this_round_loss:.3f}"
        )

        # Global Aggregation
        if self.F.weight_type == "num":
            weights = [self.data.train_examples_num_dict[client_id] for client_id in self.client_ids]
        else:
            weights = None

        serialized_parameters = self.aggregator(param_list, weights)
        self.deserialize_model(serialized_parameters)

        if should_eval:
            # TODO 启动额外程序进行推理
            self.fed_valid()

        if should_save:
            self.model_save(serialized_parameters)

        registry.register("round", self.round)
        self.model_parameters = copy.deepcopy(serialized_parameters)

    def fed_valid(self, idx=-1):
        # TODO 并行进行模型更新
        ...

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
        else:
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
                message_type=101,
                sender="0",
                receiver=list(self.comm_manager.communicators.keys()),
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

    def model_save(self, serialized_parameters=None):
        if self.phase != "train" or self.debug:
            return

        if self.F.save_valid_len:
            checkpoint_file = os.path.join(self.T.checkpoint_dir, f"round-{self.round}")
            self.deserialize_model(serialized_parameters)
            save_op = LocalSFTTrainer(
                model=self.model,
                args=self.eval_args,
            )
            save_op.save_model(checkpoint_file)
            self.logger.debug(f"Model Saved in: {checkpoint_file}")
        else:
            torch.save(self.best_glo_params, self.T.checkpoint_file)
