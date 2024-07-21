

from utils.register import registry
from trainers.LocBaseSFT import LocalSFTTrainer


@registry.register_loctrainer("centralized")
class CenLocTrainer(LocalSFTTrainer):
    ...
