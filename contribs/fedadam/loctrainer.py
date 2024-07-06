

from utils.register import registry
from trainers.LocBaseTrainer import LocalBaseTrainer


@registry.register_loctrainer("fedyogi")
@registry.register_loctrainer("fedadam")
class FedAdamLocTrainer(LocalBaseTrainer):
    ...
