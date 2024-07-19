import unittest
import torch
import grpc
from typing import Sequence
from copy import deepcopy
from commus.communicator import gRPCCommunicationManager
from commus.message import Message

"""
This test case is used to test the gRPCCommunicationManager class.
This test case contains one server communication manager and three client communication managers.
It tests several aspects of the gRPCCommunicationManager class, as follows
1. The default value of the gRPCCommunicationManager class
2. The add_communicator method of the gRPCCommunicationManager class
3. The get_communicators method of the gRPCCommunicationManager class
4. The send and receive method of the gRPCCommunicationManager class
"""


def init_comm_managers():
    TestgRPCCommunicationManager.server_comm_manager.communicators = {}
    TestgRPCCommunicationManager.client1_comm_manager.communicators = {}
    TestgRPCCommunicationManager.client2_comm_manager.communicators = {}
    TestgRPCCommunicationManager.client3_comm_manager.communicators = {}


def set_comm_managers():
    TestgRPCCommunicationManager.server_comm_manager.communicators = {
        'client1': '127.0.0.1:50052',
        'client2': '127.0.0.1:50053',
        'client3': '127.0.0.1:50054'
    }
    TestgRPCCommunicationManager.client1_comm_manager.communicators = {
        'server': '127.0.0.1:50051',
        'unknown': '127.0.0.1:50055'
    }
    TestgRPCCommunicationManager.client2_comm_manager.communicators = {
        'server': '127.0.0.1:50051'
    }
    TestgRPCCommunicationManager.client3_comm_manager.communicators = {
        'server': '127.0.0.1:50051'
    }


class TestgRPCCommunicationManager(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server_comm_manager = gRPCCommunicationManager(
            ip='127.0.0.1',
            port='50051',
            max_connection_num=3
        )
        cls.client1_comm_manager = gRPCCommunicationManager(
            ip='127.0.0.1',
            port='50052',
            max_connection_num=1
        )
        cls.client2_comm_manager = gRPCCommunicationManager(
            ip='127.0.0.1',
            port='50053',
            max_connection_num=1,
            gRPC_config={
                "grpc_max_send_message_length": 300 * 1024 * 1024,
                "grpc_max_receive_message_length": 300 * 1024 * 1024,
                "grpc_enable_http_proxy": False,
                "grpc_compression": "gzip"
            }
        )
        cls.client3_comm_manager = gRPCCommunicationManager(
            ip='127.0.0.1',
            port='50054',
            max_connection_num=1,
            gRPC_config={
                "grpc_max_send_message_length": 300 * 1024 * 1024,
                "grpc_max_receive_message_length": 300 * 1024 * 1024,
                "grpc_enable_http_proxy": False,
                "grpc_compression": "deflate"
            }
        )
        print("Create one server communication manager and three client communication managers.")

    @classmethod
    def tearDownClass(cls):
        cls.server_comm_manager.terminate_server()
        cls.client1_comm_manager.terminate_server()
        cls.client2_comm_manager.terminate_server()
        cls.client3_comm_manager.terminate_server()
        print("Terminate the server and three clients.")

    def assertNestedDataTypeEqual(self, o1, o2, places=7, msg=None, delta=None):
        if isinstance(o1, dict) and isinstance(o2, dict):
            self.assertCountEqual(list(o1.keys()), list(o2.keys()))
            for key in o1.keys():
                self.assertIn(key, o2, f"Key {key} not found in second dictionary.")
                value1, value2 = o1[key], o2[key]
                self.assertNestedDataTypeEqual(value1, value2, places=places, msg=msg, delta=delta)
        elif isinstance(o1, Sequence) and isinstance(o2, Sequence):
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

    def assertMessageEqual(self, msg1, msg2, msg=None):
        if isinstance(msg1, Message) and isinstance(msg2, Message):
            self.assertEqual(msg1.message_type, msg2.message_type)
            self.assertEqual(msg1.sender, msg2.sender)
            self.assertEqual(msg1.receiver, msg2.receiver)
            self.assertEqual(msg1.communication_round, msg2.communication_round)
            self.assertNestedDataTypeEqual(msg1.content, msg2.content)
        else:
            msg = self._formatMessage(msg, '%s == %s' % (str(msg1), str(msg1)))
            raise self.failureException(msg)

    def test_default_value(self):
        init_comm_managers()
        self.assertEqual(
            TestgRPCCommunicationManager.server_comm_manager.ip, '127.0.0.1'
        )
        self.assertEqual(
            TestgRPCCommunicationManager.server_comm_manager.port, '50051'
        )
        self.assertEqual(
            TestgRPCCommunicationManager.server_comm_manager.max_connection_num, 3
        )
        self.assertEqual(
            TestgRPCCommunicationManager.server_comm_manager.communicators, {}
        )
        self.assertEqual(
            TestgRPCCommunicationManager.server_comm_manager.gRPC_config, {
                "grpc_max_send_message_length": 300 * 1024 * 1024,
                "grpc_max_receive_message_length": 300 * 1024 * 1024,
                "grpc_enable_http_proxy": False,
                "grpc_compression": "no_compression"
            }
        )
        self.assertEqual(
            TestgRPCCommunicationManager.server_comm_manager.compression_method, grpc.Compression.NoCompression
        )

    def test_add_communicator(self):
        init_comm_managers()
        TestgRPCCommunicationManager.client1_comm_manager.add_communicator(
            'server', {'ip': '127.0.0.1', 'port': '50051'}
        )
        self.assertEqual(
            TestgRPCCommunicationManager.client1_comm_manager.communicators,
            {'server': '127.0.0.1:50051'}
        )
        TestgRPCCommunicationManager.client2_comm_manager.add_communicator(
            'server', '127.0.0.1:50051'
        )
        self.assertEqual(
            TestgRPCCommunicationManager.client2_comm_manager.communicators,
            {'server': '127.0.0.1:50051'}
        )
        TestgRPCCommunicationManager.server_comm_manager.add_communicator(
            'client1', {'ip': '127.0.0.1', 'port': '50052'}
        )
        TestgRPCCommunicationManager.server_comm_manager.add_communicator(
            'client2', '127.0.0.1:50053'
        )
        TestgRPCCommunicationManager.server_comm_manager.add_communicator(
            'client3', '127.0.0.1:50054'
        )
        self.assertEqual(
            TestgRPCCommunicationManager.server_comm_manager.communicators,
            {'client1': '127.0.0.1:50052', 'client2': '127.0.0.1:50053', 'client3': '127.0.0.1:50054'}
        )
        with self.assertRaises(TypeError):
            TestgRPCCommunicationManager.server_comm_manager.add_communicator(
                'unknown_client',
                ['127.0.0.1:50055']
            )

    def test_get_communicators(self):
        set_comm_managers()
        self.assertEqual(
            TestgRPCCommunicationManager.server_comm_manager.get_communicators(),
            {
                'client1': '127.0.0.1:50052',
                'client2': '127.0.0.1:50053',
                'client3': '127.0.0.1:50054'
            }
        )
        self.assertEqual(
            TestgRPCCommunicationManager.client1_comm_manager.get_communicators(),
            {
                'server': '127.0.0.1:50051',
                'unknown': '127.0.0.1:50055'
            }
        )
        self.assertEqual(
            TestgRPCCommunicationManager.server_comm_manager.get_communicators('client1'),
            '127.0.0.1:50052'
        )
        self.assertEqual(
            TestgRPCCommunicationManager.server_comm_manager.get_communicators(
                ['client1', 'client2']),
            {
                'client1': '127.0.0.1:50052',
                'client2': '127.0.0.1:50053'
            }
        )

    def test_send_and_receive(self):
        set_comm_managers()
        msg = Message(
            message_type=200,
            content={
                'model': {'fc1.weight': torch.ones((5, 5)), 'fc1.bias': torch.ones((5,))},
                'loss': 0.00012
            },
            communication_round=0
        )
        TestgRPCCommunicationManager.server_comm_manager.send(deepcopy(msg))
        received_msg = TestgRPCCommunicationManager.client1_comm_manager.receive()
        self.assertMessageEqual(received_msg, msg)
        received_msg = TestgRPCCommunicationManager.client2_comm_manager.receive()
        self.assertMessageEqual(received_msg, msg)
        received_msg = TestgRPCCommunicationManager.client3_comm_manager.receive()
        self.assertMessageEqual(received_msg, msg)
        TestgRPCCommunicationManager.server_comm_manager.send(deepcopy(msg), ['client1', 'client2'])
        received_msg = TestgRPCCommunicationManager.client1_comm_manager.receive()
        self.assertMessageEqual(received_msg, msg)
        received_msg = TestgRPCCommunicationManager.client2_comm_manager.receive()
        self.assertMessageEqual(received_msg, msg)
        TestgRPCCommunicationManager.client1_comm_manager.send(deepcopy(msg), 'server')
        received_msg = TestgRPCCommunicationManager.server_comm_manager.receive()
        self.assertMessageEqual(received_msg, msg)
        with self.assertRaises(ConnectionError):
            TestgRPCCommunicationManager.client1_comm_manager.send(deepcopy(msg), 'unknown')


if __name__ == '__main__':
    unittest.main()
