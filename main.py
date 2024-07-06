"""
Fine-tuning the library models for sequence to sequence.
"""
from configs import build_config
from utils.general import setup_imports
from utils.register import registry


def main():
    setup_imports()

    config = build_config()

    trainer = registry.get_fedtrainer(config.federated_config.fl_algorithm)()
    trainer.run()


if __name__ == "__main__":
    main()
