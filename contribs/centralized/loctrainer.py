

from utils.register import registry
from trainers.LocBaseTrainer import LocalBaseTrainer


@registry.register_loctrainer("centralized")
class CenLocTrainer(LocalBaseTrainer):
    ...
