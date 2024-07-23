import unittest
import pickle
import os
from utils.misc import build_prompt, build_two_prompt, build_legal_prompt, exact_matching, reuse_results, prompt_cases


class MiscTest(unittest.TestCase):
    def test_build_legal_prompt(self):
        self.assertEqual(
            build_legal_prompt("lcp"), "现在你是一个法律专家，请你判断下面案件的类型。\n案件：{}\n案件类型是："
        )

    def test_build_prompt(self):
        self.assertEqual(
            build_prompt(prompt_cases[1], "text"),
            "现在你是一个法律专家，请你给出下面案件的类型。\n案件：text\n案件类型是："
        )
        self.assertEqual(
            build_prompt(prompt_cases[2], "text"),
            "现在你是一个法律专家，请你给出下面案件的类型。\n案件：text\n案件类型："
        )

    def test_build_two_prompt(self):
        self.assertEqual(
            build_two_prompt(prompt_cases[1], "text"),
            "现在你是一个法律专家，请你给出下面案件的类型。\n案件：text\n案件类型是：\n让我们一步一步思考。"
        )

    def test_extract_matching(self):
        self.assertAlmostEqual(
            exact_matching(["text case", "test case"], ["text case", "test case"]), 0.0
        )
        self.assertAlmostEqual(
            exact_matching("text file", "text case"), 0.667
        )
        self.assertAlmostEqual(
            exact_matching("test", "text"), 0.75
        )
        self.assertAlmostEqual(
            exact_matching("test", "unknown"), 0.0
        )

    def test_reuse_results(self):
        old_results = {"prompt1": "1", "result": [
            ["text1", "predict_label1", "predict_label1"],
            ["text2", "predict_label2", "predict_label2"],
            ["text3", "predict_label3", "predict_label3"],
            ["text4", "predict_label4", "predict_label4"],
            ["text5", "predict_label5", "predict_label5"],
            ["text6", "predict_label6", "predict_label6"],
            ["text7", "predict_label7", "predict_label7"],
            ["text8", "predict_label8", "predict_label8"],
            ["text9", "label9", "predict_label9"],
            ["text10", "label10", "predict_label10"]
        ]}
        new_results = {
            'prompt': '', 'result': [
                ('texts', 'labels', 'predict_labels'),
                ('text1', 'predict_label1', 'predict_label1'),
                ('text2', 'predict_label2', 'predict_label2'),
                ('text3', 'predict_label3', 'predict_label3'),
                ('text4', 'predict_label4', 'predict_label4'),
                ('text5', 'predict_label5', 'predict_label5'),
                ('text6', 'predict_label6', 'predict_label6'),
                ('text7', 'predict_label7', 'predict_label7')
            ]
        }
        with open("results.pkl", "wb") as f:
            pickle.dump(old_results, f)
        results = {"prompt": "", "result": [("texts", "labels", "predict_labels")]}
        self.assertEqual(
            reuse_results(
                "text case", results),
            (-1, results)
        )
        self.assertEqual(
            reuse_results(
                "results.pkl", results),
            (1, new_results)
        )
        os.remove("results.pkl")


if __name__ == '__main__':
    unittest.main()
