import grpc
from commus.message import Message
import commus.gRPC_comm_manager_pb2
import commus.gRPC_comm_manager_pb2_grpc
from commus.communicator import gRPCCommManager

import sys
import time
from loguru import logger

role = str(sys.argv[1])
server_ip = "172.17.0.3"
server_port = "15001"
client_ip = "172.17.0.3"
client_port_1 = "15002"
client_port_2 = "15003"
client_name_1 = 0
client_name_2 = 1
num_sub = 2

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
