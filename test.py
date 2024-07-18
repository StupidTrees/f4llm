
import grpc
from commus.message import Message
import commus.gRPC_communication_manager_pb2
import commus.gRPC_communication_manager_pb2_grpc
from commus.communicator import gRPCCommunicationManager

import sys
import time
from loguru import logger

import torch


from torch import nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        # self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        # x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 启动命令 bash ./scripts/test.sh
role = str(sys.argv[1])
server_ip = client_ip = "127.0.0.1"  # docker hostname -I
server_port = "15001"
client_port_1 = "15002"
client_port_2 = "15003"
client_name_1 = "0"
client_name_2 = "1"
num_sub = 2

# 检测内容
# 1) 单数据无bug通信稳定复现；
# 2) 多数据无bug通信稳定复现；
# 3) 大数据无bug通信稳定复现, e.g., 20MB tensor
shape = (512, 512, 20)
# tensor = torch.randn(shape)
model = MLP(784, 10, 10)
tensor = model.state_dict()

if role == "server":
    logger.info("server start")
    comm_manager = gRPCCommunicationManager(
        ip=server_ip,
        port=server_port,
        max_connection_num=num_sub
    )
    client_num = 0
    while client_num < num_sub:
        msg = comm_manager.receive()
        if msg.message_type == 100:
            client_num += 1
            comm_manager.add_communicators(
                communicator_id=msg.sender,
                communicator_address=f"{msg.content['client_ip']}:{msg.content['client_port']}")
            logger.info(f"Subserver {msg.sender} joined in.")
            logger.info(comm_manager.communicators.keys())
    logger.debug("all subserver connect")

    for i in range(5):
        print(f"{role} {i}")
        time.sleep(2)

    client_num = 0
    while client_num < num_sub:
        # logger.info(f"server waiting...")
        msg = comm_manager.receive()
        if msg.message_type == 200:
            client_num += 1
            logger.critical(f'{msg.sender} context {msg.content["model"]}')
    logger.debug(f"{role} done !!!")
    comm_manager.terminate_server()
    print(f"{role} terminate")

elif role == "client_1":
    time.sleep(3)
    logger.info(f"client {role} start")
    comm_manager = gRPCCommunicationManager(
        ip=client_ip,
        port=client_port_1,
        max_connection_num=1
    )
    comm_manager.add_communicators(
        communicator_id=server_ip,
        communicator_address='{}:{}'.format(server_ip, server_port)
    )
    comm_manager.send(
        Message(
            message_type=100,
            sender=client_name_1,
            receiver=[server_ip],
            content={
                'client_ip': client_ip,
                'client_port': client_port_1
            }
        )
    )
    for i in range(10):
        print(f"{role} {i}")
        time.sleep(1)
    logger.debug(f"{role} done !!!")

    comm_manager.send(
        Message(
            message_type=200,
            sender=client_name_1,
            receiver=[server_ip],
            content={
                'model': tensor,
            }
        )
    )
    time.sleep(3)
    comm_manager.terminate_server()
    print(f"{role} terminate")

else:
    time.sleep(3)
    logger.info(f"client {role} start")
    comm_manager = gRPCCommunicationManager(
        ip=client_ip,
        port=client_port_2,
        max_connection_num=1,
    )
    comm_manager.add_communicators(
        communicator_id=server_ip,
        communicator_address='{}:{}'.format(server_ip, server_port)
    )
    comm_manager.send(
        Message(
            message_type=100,
            sender=client_name_2,
            receiver=[server_ip],
            content={
                'client_ip': client_ip,
                'client_port': client_port_2
            }
        )
    )
    for i in range(10):
        print(f"{role} {i}")
        time.sleep(0.5)
    logger.debug(f"{role} done !!!")

    comm_manager.send(
        Message(
            message_type=200,
            sender=client_name_2,
            receiver=[server_ip],
            content={
                'model': tensor,
            }
        )
    )
    time.sleep(1)
    comm_manager.terminate_server()
    print(f"{role} terminate")
