"""
Fine-tuning the library models for sequence to sequence.
"""
from configs import build_config
from utils.general import setup_imports
from utils.register import registry


def run_experiment(config):
    phase = registry.get("phase")
    if phase == "train":
        engine = registry.get_fedtrainer(config.federated_config.fl_algorithm)()
    else:
        engine = registry.get_fedtrainer(config.federated_config.fl_algorithm)()
    engine.run()


def main():
    setup_imports()

    config = build_config()

    run_experiment(config)


if __name__ == "__main__":
    main()
