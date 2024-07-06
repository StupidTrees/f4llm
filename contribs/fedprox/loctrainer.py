
import torch
from utils.register import registry
from trainers.LocBaseTrainer import LocalBaseTrainer


@registry.register_loctrainer("fedprox")
class FedProxLocTrainer(LocalBaseTrainer):
    def __init__(self, global_state, prox_mu, **kwargs):
        super(FedProxLocTrainer, self).__init__(**kwargs)
        self.global_state = global_state
        self.mu = prox_mu

    def compute_loss(self, model, inputs, return_outputs=False):

        return_values = super(FedProxLocTrainer, self).compute_loss(model, inputs, return_outputs=return_outputs)

        if return_outputs:
            loss, outputs = return_values
        else:
            loss = return_values

        # Apply FedProx Loss
        for name, param in model.named_parameters():
            name = name.replace(".default", "")  # TODO: May need changes. to accord with peft
            name = name.replace("module.", "")
            # only trainable parameters
            if not param.requires_grad:
                continue
            else:
                loss += self.mu / 2 * torch.norm(param - self.global_state[name]) ** 2

        return (loss, outputs) if return_outputs else loss
