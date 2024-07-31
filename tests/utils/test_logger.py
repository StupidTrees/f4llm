import unittest
import argparse
import sys
from loguru import logger
from utils.logger import formatter, setup_logger


class TestLogger(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument("--local_rank", type=int, default=0)
        parser.add_argument("--save_dir", type=str, default="logs")
        cls.training_args = parser.parse_args()

    def test_formatter(self):
        logger.add(
            sys.stderr, format=formatter,
            colorize=True, enqueue=True
        )
        logger.info("Logging setup complete.")
        logger.critical("Logging setup complete.")

    def test_setup_logger_main_process(self):
        set_up_ed_logger = setup_logger(self.training_args)
        self.assertIsNotNone(set_up_ed_logger)

    def test_setup_logger_sub_process(self):
        self.training_args.local_rank = 1
        set_up_ed_logger = setup_logger(self.training_args)
        self.assertIsNotNone(set_up_ed_logger)
        self.training_args.local_rank = 0


if __name__ == '__main__':
    unittest.main()
