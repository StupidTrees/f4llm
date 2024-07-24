import copy
import os
from copy import deepcopy

from peft import (
    get_peft_model_state_dict,
    set_peft_model_state_dict)

from utils.register import registry
from datas.base_data import FedBaseDataset
from utils.general import is_petuning, metric_save
from utils.serialization import SerializationTool


class BaseEngine(object):
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
        self.logger.info(f"{self.role} building dataset ...")
        self.data = registry.get_data_class(self.D.data_name)()
        self.data.load_data()

    def _build_model(self):
        self.logger.info(f"{self.role} building model ...")
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

    def run(self):
        raise NotImplementedError

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
            model_parameters = deepcopy(get_peft_model_state_dict(self.model))
        else:
            model_parameters = SerializationTool.serialize_model(self.model)
        return model_parameters

    def deserialize_model(self, serialized_parameters):
        if is_petuning(self.M.tuning_type):
            set_peft_model_state_dict(self.model, serialized_parameters)
        else:
            SerializationTool.deserialize_model(self.model, serialized_parameters)

    def metric_save(self):
        if self.debug:
            return
        metric_save(self, self.T, self.logger)

    def build_eval_cmd(self):
        base_opts = [
            "main.py", "--do_eval",
            "--raw_dataset_path", self.D.raw_dataset_path,
            "--partition_dataset_path", self.D.partition_dataset_path,
            "--model_name_or_path", self.M.model_name_or_path,
            "--output_dir", os.path.dirname(self.T.output_dir),
            "--model_type", self.M.model_type,
            "--task_name", self.D.task_name,
            "--fl_algorithm", self.F.fl_algorithm,
            "--config_path", self.T.config_path,
            "--role", "client",
            "--times", self.T.times
        ]
        return base_opts

    @property
    def role(self):
        return self.F.role
