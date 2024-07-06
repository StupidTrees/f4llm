

from utils.register import registry
from trainers.LocBaseTrainer import LocalBaseTrainer

from trl import DPOTrainer


@registry.register_loctrainer("fedavg")
class FedAvgSFTLocTrainer(LocalBaseTrainer):
    ...


@registry.register_loctrainer("fedavg_dpo")
class FedAvgDPOLocTrainer(DPOTrainer):
    ...
