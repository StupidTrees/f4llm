
import grpc
from commus.message import Message
import commus.gRPC_comm_manager_pb2
import commus.gRPC_comm_manager_pb2_grpc
from commus.communicator import gRPCCommManager

import sys
import time
from loguru import logger

import torch

role = str(sys.argv[1])
server_ip = client_ip = "172.17.0.2"  # docker hostname -I
server_port = "15001"
client_port_1 = "15002"
client_port_2 = "15003"
client_name_1 = 0
client_name_2 = 1
num_sub = 2

# 检测内容
# 1) 单数据无bug通信稳定复现；
# 2) 多数据无bug通信稳定复现；
# 3) 大数据无bug通信稳定复现, e.g., 20MB tensor
# shape = (512, 512, 20)
# tensor = torch.randn(shape)

if role == "server":
    logger.info("server start")
    comm_manager = gRPCCommManager(
        host=server_ip,
        port=server_port,
        client_num=num_sub
    )
    client_num = 0
    while client_num < num_sub:
        msg = comm_manager.receive()
        if msg.msg_type == "join_in":
            client_num += 1
            comm_manager.add_neighbors(neighbor_id=msg.sender,
                                       address=f"{msg.content['client_ip']}:{msg.content['client_port']}")
            logger.info(f"Subserver {msg.sender} joined in.")
            logger.info(comm_manager.neighbors.keys())
    logger.debug("all subserver connect")

    for i in range(5):
        print(f"{role} {i}")
        time.sleep(2)

    client_num = 0
    while client_num < num_sub:
        # logger.info(f"server waiting...")
        msg = comm_manager.receive()
        if msg.msg_type == "param":
            client_num += 1
            logger.critical(f'{msg.sender} context {msg.content["data"]}')
    logger.debug(f"{role} done !!!")

elif role == "client_1":
    time.sleep(3)
    logger.info(f"client {role} start")
    comm_manager = gRPCCommManager(
        host=client_ip,
        port=client_port_1,
        client_num=1,
        cfg=None
    )
    comm_manager.add_neighbors(
        neighbor_id=server_ip,
        address='{}:{}'.format(server_ip, server_port)
    )
    comm_manager.send(
        Message(
            msg_type='join_in',
            sender=client_name_1,
            receiver=[server_ip],
            timestamp=0,
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
            msg_type='param',
            sender=client_name_1,
            receiver=[server_ip],
            timestamp=0,
            content={
                'data': [4, 5, 6],
            }
        )
    )

else:
    time.sleep(3)
    logger.info(f"client {role} start")
    comm_manager = gRPCCommManager(
        host=client_ip,
        port=client_port_2,
        client_num=1,
        cfg=None
    )
    comm_manager.add_neighbors(
        neighbor_id=server_ip,
        address='{}:{}'.format(server_ip, server_port)
    )
    comm_manager.send(
        Message(
            msg_type='join_in',
            sender=client_name_2,
            receiver=[server_ip],
            timestamp=0,
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
            msg_type='param',
            sender=client_name_2,
            receiver=[server_ip],
            timestamp=0,
            content={
                'data': [1, 2, 3],
            }
        )
    )
