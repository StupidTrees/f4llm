import random
import time

import grpc
from . import gRPC_communication_manager_pb2_grpc as gRPC_communication_manager_pb2_grpc
from concurrent import futures
from .gRPC_server import gRPCComServeFunc
from .message import Message


class gRPCCommunicationManager:
    def __init__(
            self,
            ip: str = "127.0.0.1",
            port: str = "50051",
            max_connection_num: int = 1,
            gRPC_config=None
    ):
        if gRPC_config is None:
            gRPC_config = {
                "grpc_max_send_message_length": 300 * 1024 * 1024,
                "grpc_max_receive_message_length": 300 * 1024 * 1024,
                "grpc_enable_http_proxy": False,
                "grpc_compression": "no_compression"
            }
        self._ip = ip
        self._port = port
        self._max_connection_num = max_connection_num
        self._gRPC_config = gRPC_config
        self.server_funcs = gRPCComServeFunc()
        options = [
            ("grpc.max_send_message_length", gRPC_config["grpc_max_send_message_length"]),
            ("grpc.max_receive_message_length", gRPC_config["grpc_max_receive_message_length"]),
            ("grpc.enable_http_proxy", gRPC_config["grpc_enable_http_proxy"]),
        ]
        if gRPC_config["grpc_compression"].lower() == 'deflate':
            self._compression_method = grpc.Compression.Deflate
        elif gRPC_config["grpc_compression"].lower() == 'gzip':
            self._compression_method = grpc.Compression.Gzip
        else:
            self._compression_method = grpc.Compression.NoCompression
        self._gRPC_server = self.serve(
            max_workers=max_connection_num,
            ip=ip,
            port=port,
            options=options
        )
        self._communicators = dict()

    @property
    def ip(self):
        return self._ip

    @ip.setter
    def ip(self, value):
        self._ip = value

    @property
    def port(self):
        return self._port

    @port.setter
    def port(self, value):
        self._port = value

    @property
    def max_connection_num(self):
        return self._max_connection_num

    @max_connection_num.setter
    def max_connection_num(self, value):
        self._max_connection_num = value

    @property
    def gRPC_config(self):
        return self._gRPC_config

    @gRPC_config.setter
    def gRPC_config(self, value):
        self._gRPC_config = value

    @property
    def compression_method(self):
        return self._compression_method

    @compression_method.setter
    def compression_method(self, value):
        self._compression_method = value

    @property
    def communicators(self):
        return self._communicators

    def serve(self, max_workers: int, ip: str, port: str, options: dict | None):
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=max_workers),
            compression=self._compression_method,
            options=options
        )
        gRPC_communication_manager_pb2_grpc.add_gRPCComServeFuncServicer_to_server(
            self.server_funcs, server)
        server.add_insecure_port("{}:{}".format(ip, port))
        server.start()

        return server

    def terminate_server(self):
        self._gRPC_server.stop(grace=None)

    def add_communicators(self, communicator_id: str, communicator_address: dict | str):
        if isinstance(communicator_address, dict):
            self._communicators[communicator_id] = f"{communicator_address['ip']}:{communicator_address['port']}"
        elif isinstance(communicator_address, str):
            self._communicators[communicator_id] = communicator_address
        else:
            raise TypeError(f"The type of communicator_address ({type(communicator_address)}) is not supported")

    def get_communicators(self, communicator_id: str | list | None):
        address = dict()
        if communicator_id:
            if isinstance(communicator_id, list):
                for each_communicator in communicator_id:
                    address[each_communicator] = self.get_communicators(each_communicator)
                return address
            else:
                return self._communicators[communicator_id]
        else:
            return self._communicators

    def _create_stub(self, receiver_address: str):
        channel = grpc.insecure_channel(receiver_address,
                                        compression=self.compression_method,
                                        # options=(('grpc.enable_http_proxy',
                                        #           0),)
                                        )
        stub = gRPC_communication_manager_pb2_grpc.gRPCComServeFuncStub(channel)
        return stub, channel

    def _send(self, receiver_address: str, message: Message, max_retry: int = 3):
        request = message.transform(to_list=True)
        attempts = 0
        retry_interval = 1
        success_flag = False
        while attempts < max_retry:
            stub, channel = self._create_stub(receiver_address)
            try:
                stub.sendMessage(request)
                channel.close()
                success_flag = True
            except grpc._channel._InactiveRpcError as error:
                print(error)
                attempts += 1
                time.sleep(retry_interval)
                retry_interval *= 2
            finally:
                channel.close()
            if success_flag:
                break

    def send(self, message: Message, receiver: str | list | None = None):
        if receiver is not None:
            if not isinstance(receiver, list):
                receiver = [receiver]
            for each_receiver in receiver:
                if each_receiver in self._communicators.keys():
                    receiver_address = self._communicators[each_receiver]
                    self._send(receiver_address, message)
        else:
            for each_receiver in self._communicators.keys():
                receiver_address = self._communicators[each_receiver]
                self._send(receiver_address, message)

    def receive(self):
        received_message = self.server_funcs.receive()
        message = Message()
        message.parse(received_message.msg)
        return message
