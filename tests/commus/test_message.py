import unittest
import numpy as np
import torch
from copy import deepcopy
from commus.message import Message


"""
This test case is used to test the Message class in the communication module.
It tests several aspects of the Message class, as follows:
1. The transform method of the Message class
1.1 Transform int type message to Request
1.2 Transform float type message to Request
1.3 Transform str type message to Request
1.4 Transform list type message to Request
1.5 Transform tuple type message to Request
1.6 Transform dict type message to Request
1.7 Transform numpy array type message to Request
1.8 Transform torch tensor type message to Request

2. The parse method of the Message class
2.1 Parse int type message
2.2 Parse float type message
2.3 Parse str type message
2.4 Parse list type message
2.5 Parse tuple type message
2.6 Parse dict type message
2.7 Parse numpy array type message
2.8 Parse torch tensor type message

3. The __lt__ method of the Message class
4. The default value of the Message class
5. The count_bytes method of the Message class 
"""


class TestMessage(unittest.TestCase):
    def assertNestedDataTypeEqual(self, o1, o2, places=7, msg=None, delta=None):
        if isinstance(o1, dict) and isinstance(o2, dict):
            self.assertCountEqual(list(o1.keys()), list(o2.keys()))
            for key in o1.keys():
                self.assertIn(key, o2, f"Key {key} not found in second dictionary.")
                value1, value2 = o1[key], o2[key]
                self.assertNestedDataTypeEqual(value1, value2, places=places, msg=msg, delta=delta)
        elif (isinstance(o1, list) or isinstance(o1, tuple)) and (isinstance(o2, list) or isinstance(o2, tuple)):
            self.assertEqual(len(o1), len(o2))
            length = len(o1)
            for idx in range(length):
                self.assertNestedDataTypeEqual(o1[idx], o2[idx], places=places, msg=msg, delta=delta)
        elif isinstance(o1, int) and isinstance(o2, int):
            self.assertEqual(o1, o2, msg=msg)
        elif isinstance(o1, str) and isinstance(o2, str):
            self.assertEqual(o1, o2, msg=msg)
        elif isinstance(o1, float) and isinstance(o2, float):
            self.assertAlmostEqual(o1, o2, places=places, msg=msg, delta=delta)
        elif hasattr(o1, 'tolist') and hasattr(o2, 'tolist'):
            self.assertTrue((o1 == o2).all(), msg=msg)
        else:
            msg = self._formatMessage(msg, '%s == %s' % (str(o1), str(o2)))
            raise self.failureException(msg)

    def test_transform_and_parse(self):
        msg = Message(
            message_type=200,
            sender='0',
            receiver='1',
            content=100,
            communication_round=0
        )
        received_msg = Message()

        msg.content = 100
        request = msg.transform(to_list=True)
        received_msg.parse(request.msg)
        self.assertEqual(received_msg.content, msg.content)

        msg.content = 0.00001
        request = msg.transform(to_list=True)
        received_msg.parse(request.msg)
        self.assertAlmostEqual(received_msg.content, msg.content)

        msg.content = "100"
        request = msg.transform(to_list=True)
        received_msg.parse(request.msg)
        self.assertEqual(received_msg.content, msg.content)

        msg.content = [1, 2, 3]
        request = msg.transform(to_list=True)
        received_msg.parse(request.msg)
        self.assertEqual(received_msg.content, msg.content)

        msg.content = (1, 2, 3)
        request = msg.transform(to_list=True)
        received_msg.parse(request.msg)
        self.assertSequenceEqual(received_msg.content, msg.content)

        msg.content = [(1, 2, 3), (2, 3, 4, 5), (3, 4, 5, 6)]
        request = msg.transform(to_list=True)
        received_msg.parse(request.msg)
        self.assertNestedDataTypeEqual(received_msg.content, msg.content)

        msg.content = {0: [1, 2], 1: 1, 2: 0.00001, 3: "test"}
        request = msg.transform(to_list=True)
        received_msg.parse(request.msg)
        self.assertNestedDataTypeEqual(received_msg.content, msg.content)

        msg.content = {'data': [1, 2, 3], 'loss': 0.0001, 'other': 1}
        request = msg.transform(to_list=True)
        received_msg.parse(request.msg)
        self.assertNestedDataTypeEqual(received_msg.content, msg.content)

        msg.content = {'model': np.ones((5, 5))}
        request = msg.transform(to_list=True)
        received_msg.parse(request.msg)
        self.assertNestedDataTypeEqual(received_msg.content, {'model': np.ones((5, 5))})

        msg.content = {'model': torch.ones((5, 5))}
        request = msg.transform(to_list=True)
        received_msg.parse(request.msg)
        self.assertNestedDataTypeEqual(received_msg.content, {'model': torch.ones((5, 5))})

        msg.content = {'model': {'fc1.weight': torch.ones((5, 5)), 'fc1.bias': torch.ones((5,))}}
        request = msg.transform(to_list=True)
        received_msg.parse(request.msg)
        self.assertNestedDataTypeEqual(
            received_msg.content, {'model': {'fc1.weight': torch.ones((5, 5)), 'fc1.bias': torch.ones((5,))}}
        )

        content = {
            'data': [1, 2, 3],
            'model': {'fc1.weight': torch.ones((5, 5)), 'fc1.bias': torch.ones((5,))},
            'loss': 0.0001,
            'other': {0: [1, 2, 3], 1: 0.0001, 2: {'data': (3.0, 4.0, 0.1), 'loss': 0.09}}
        }
        msg.content = deepcopy(content)
        request = msg.transform(to_list=True)
        received_msg.parse(request.msg)
        self.assertNestedDataTypeEqual(received_msg.content, content)

        content = {
            'data': [1, 2, 3],
            'model': {
                'original_model': {'fc1.weight': torch.ones((5, 5)), 'fc1.bias': torch.ones((5,))},
                'proxy_param': {
                    0: {'fc1.weight': torch.ones((5, 5)), 'fc1.bias': torch.ones((5,))},
                    1: {'fc1.weight': torch.ones((5, 5)), 'fc1.bias': torch.ones((5,))},
                    2: {'fc1.weight': torch.ones((5, 5)), 'fc1.bias': torch.ones((5,))}
                },
                'ds': [torch.ones((5, 5)), torch.ones((5, 5))],
                'float': 0.09,
                'int': 1
            },
            'loss': 0.0001,
            'other': {0: [1, 2, 3], 1: 0.0001, 2: {'data': (3.0, 4.0, 0.1), 'loss': 0.09}}
        }
        msg.content = deepcopy(content)
        request = msg.transform(to_list=True)
        received_msg.parse(request.msg)
        self.assertNestedDataTypeEqual(received_msg.content, content)

        msg.content = None
        with self.assertRaises(ValueError):
            msg.transform(to_list=True)

    def test_lt(self):
        msg1 = Message(
            message_type=200,
            sender='0',
            receiver='1',
            content=100,
            communication_round=0
        )
        msg2 = Message(
            message_type=200,
            sender='0',
            receiver='1',
            content=100,
            communication_round=1
        )
        self.assertTrue(msg1 < msg2)
        msg1.timestamp = 1
        msg2.timestamp = 1
        self.assertTrue(msg1 < msg2)

    def test_default_value(self):
        msg = Message()
        self.assertEqual(msg.message_type, -1)
        self.assertEqual(msg.sender, '-1')
        self.assertEqual(msg.receiver, '-1')
        self.assertEqual(msg.content, '')
        self.assertEqual(msg.communication_round, 0)

    def test_count_bytes(self):
        msg = Message(
            receiver=['client1', 'client2', 'client3'],
            content='test',

        )
        self.assertEqual(msg.count_bytes(), (56, 168))


if __name__ == '__main__':
    unittest.main()
