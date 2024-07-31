import unittest
import os
import argparse
import torch
import torch.nn as nn
from utils.logger import setup_logger
from utils.general import (
    pickle_read, pickle_write, read_json, write_json, file_write, make_sure_dirs,
    rm_dirs, rm_file, get_cpus, get_memory_usage, LoadBalanceSampling, cosine_learning_rate,
    custom_pad_sequence, get_parameter_number, is_petuning, get_peft_parameters, setup_seed,
    is_best, run_process, end_log, metric_save, setup_imports
)


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class GeneralTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument("--local_rank", type=int, default=0)
        parser.add_argument("--save_dir", type=str, default="logs")
        parser.add_argument("--metric_log_file", type=str, default="test_metric.json")
        parser.add_argument("--metric_file", type=str, default="test_metric.json")
        cls.training_args = parser.parse_args()
        parser = argparse.ArgumentParser()
        parser.add_argument("--metric_log", type=str, default="test")
        cls.trainer = parser.parse_args()

    def test_pickle_write_and_read_and_remove(self):
        data = {"key": "value", "key2": "value2"}
        pickle_write(data, "pickle_test.pkl", "wb")
        self.assertEqual(pickle_read("pickle_test.pkl", "rb"), data)
        rm_file("pickle_test.pkl")

    def test_json_write_and_read_and_remove(self):
        data = {"key": "value", "key2": "value2"}
        write_json(data, "json_test.json")
        self.assertEqual(read_json("json_test.json"), data)
        rm_file("json_test.json")
        write_json(data, "json_test.jsonl")
        """Note that this function is not implemented yet"""
        self.assertEqual(read_json("json_test.jsonl"), ['key', 'key2'])
        rm_file("json_test.jsonl")

    def test_file_write_and_remove(self):
        file_write("test", "test.txt", "w")
        with open("test.txt", "r") as f:
            self.assertEqual(f.readline(), "test\n")
        rm_file("test.txt")

    def test_dirs_make_and_remove(self):
        make_sure_dirs("test", role="client")
        self.assertFalse("test" in os.listdir())
        make_sure_dirs("test")
        self.assertTrue("test" in os.listdir())
        rm_dirs("test")
        self.assertFalse("test" in os.listdir())

    def test_get_cpus(self):
        self.assertTrue(get_cpus() > 0)

    def test_memory_usage(self):
        self.assertTrue(get_memory_usage() > 0)

    def test_LoadBalanceSampling(self):
        data = [1, 2, 3, 4, 5]
        lbs = LoadBalanceSampling(data, 2)
        self.assertEqual(len(lbs), 2)
        self.assertEqual(len(lbs[0]), 3)
        self.assertEqual(len(lbs[1]), 2)

        data = [1, 2, 3, 4, 5, 6]
        lbs = LoadBalanceSampling(data, 3)
        self.assertEqual(len(lbs), 3)
        self.assertEqual(len(lbs[0]), 2)
        self.assertEqual(len(lbs[1]), 2)
        self.assertEqual(len(lbs[2]), 2)

    def test_cosine_learning_rate(self):
        self.assertAlmostEqual(
            cosine_learning_rate(1, 10, 0.1, 0.1), 0.1
        )
        self.assertAlmostEqual(
            cosine_learning_rate(1, 10, 0.1, 0), 0.09755282581475769
        )

    def test_custom_pad_sequence(self):
        data = [torch.ones((10,)), torch.ones((5,))]
        self.assertTrue(
            torch.equal(custom_pad_sequence(data, padding_value=-100),
                        torch.tensor([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                                      [-100., -100., -100., -100., -100., 1., 1., 1., 1., 1.]]))
        )
        self.assertTrue(
            torch.equal(custom_pad_sequence(data, padding_value=-100, left_padding=False),
                        torch.tensor([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                                      [1., 1., 1., 1., 1., -100., -100., -100., -100., -100.]]))
        )

    def test_get_parameter_number(self):
        model = Net(784, 250, 10)
        self.assertAlmostEqual(
            get_parameter_number(model)['Total'], 0.1988
        )
        self.assertAlmostEqual(
            get_parameter_number(model)['Trainable'], 0.1988
        )

    def test_is_petuning(self):
        self.assertTrue(is_petuning(["lora", "adapter", "bitfit", "prefix", "p-tuning"]))
        self.assertTrue(is_petuning(['lora']))
        self.assertTrue(is_petuning('lora'))
        self.assertFalse(is_petuning('petuning'))

    def test_setup_seed(self):
        setup_seed(42)
        self.assertEqual(torch.initial_seed(), 42)

    def test_is_best(self):
        self.assertTrue(is_best(1, 2, False))
        self.assertFalse(is_best(2, 1, False))
        self.assertFalse(is_best(1, 2, True))
        self.assertTrue(is_best(2, 1, True))

    def test_metric_save(self):
        logger = setup_logger(self.training_args)
        metric_save(self.trainer, self.training_args, logger=logger)
        rm_file("test_metric.json")

    def test_run_process(self):
        with self.assertRaises(AssertionError):
            self.assertNotEqual(run_process("None"), None)

    def test_imports(self):
        setup_imports()
        self.assertTrue("torch" in globals())
        setup_imports()


if __name__ == '__main__':
    unittest.main()
