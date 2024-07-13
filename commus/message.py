import json
import pickle
import base64
import numpy as np
from . import gRPC_communication_manager_pb2 as gRPC_communication_manager_pb2
from datetime import datetime
from pympler import asizeof


"""
Note that:
message_type:
    100: build federated learning connection
    101: end federated learning connection
    200: update model parameters, the content should contain a key 'model' with the value of model parameters
    300: transfer metrics
"""


def b64serializer(x):
    return base64.b64encode(pickle.dumps(x))


class Message(object):
    """
    The data exchanged during an FL course are abstracted as 'Message'.
    A message object includes:
        message_type: The type of message, which is used to trigger the
        corresponding handlers of server/client
        sender: The sender's ID
        receiver: The receiver's ID
        communication_round: The training round of the message, which is determined by
        the sender and used to filter out the outdated messages.
    """
    def __init__(
            self,
            message_type: int = -1,
            sender: str = "-1",
            receiver: str | list = "-1",
            content: str | dict = "",
            communication_round: int = 0
    ):
        self._message_type = message_type
        self._sender = sender
        self._receiver = receiver
        self._content = content
        self._communication_round = communication_round
        self._timestamp = datetime.now().timestamp()
        self.param_serializer = b64serializer

    @property
    def message_type(self):
        return self._message_type

    @message_type.setter
    def message_type(self, value):
        self._message_type = value

    @property
    def sender(self):
        return self._sender

    @sender.setter
    def sender(self, value):
        self._sender = value

    @property
    def receiver(self):
        return self._receiver

    @receiver.setter
    def receiver(self, value):
        self._receiver = value

    @property
    def content(self):
        return self._content

    @content.setter
    def content(self, value):
        self._content = value

    @property
    def communication_round(self):
        return self._communication_round

    @communication_round.setter
    def communication_round(self, value):
        self._communication_round = value

    @property
    def timestamp(self):
        return self._timestamp

    @timestamp.setter
    def timestamp(self, value):
        self._timestamp = value

    def __lt__(self, other):
        if self.timestamp != other.timestamp:
            return self.timestamp < other.timestamp
        else:
            return self.communication_round < other.communication_round

    def create_by_type(self, value, nested=False):
        if isinstance(value, dict):
            if isinstance(list(value.keys())[0], str):
                m_dict = gRPC_communication_manager_pb2.mDict_keyIsString()
                key_type = 'string'
            else:
                m_dict = gRPC_communication_manager_pb2.mDict_keyIsInt()
                key_type = 'int'
            for key in value.keys():
                m_dict.dict_value[key].MergeFrom(
                    self.create_by_type(value[key], nested=True))
            if nested:
                msg_value = gRPC_communication_manager_pb2.MsgValue()
                if key_type == 'string':
                    msg_value.dict_msg_string_key.MergeFrom(m_dict)
                else:
                    msg_value.dict_msg_int_key.MergeFrom(m_dict)
                return msg_value
            else:
                return m_dict
        elif isinstance(value, list) or isinstance(value, tuple):
            m_list = gRPC_communication_manager_pb2.mList()
            for each in value:
                m_list.list_value.append(self.create_by_type(each,
                                                             nested=True))
            if nested:
                msg_value = gRPC_communication_manager_pb2.MsgValue()
                msg_value.list_msg.MergeFrom(m_list)
                return msg_value
            else:
                return m_list
        else:
            m_single = gRPC_communication_manager_pb2.mSingle()
            if type(value) in [int, np.int32]:
                m_single.int_value = value
            elif type(value) in [str, bytes]:
                m_single.str_value = value
            elif type(value) in [float, np.float32]:
                m_single.float_value = value
            else:
                raise ValueError(
                    f'The data type {type(value)} has not been supported.')

            if nested:
                msg_value = gRPC_communication_manager_pb2.MsgValue()
                msg_value.single_msg.MergeFrom(m_single)
                return msg_value
            else:
                return m_single

    def transform_to_list(self, x):
        if isinstance(x, list) or isinstance(x, tuple):
            return [self.transform_to_list(each_x) for each_x in x]
        elif isinstance(x, dict):
            for key in x.keys():
                x[key] = self.transform_to_list(x[key])
            return x
        else:
            if hasattr(x, 'tolist'):
                # if self.msg_type == 'model_para':
                return self.param_serializer(x)
                # else:
                # return x.tolist()
            else:
                return x

    def build_msg_value(self, value):
        msg_value = gRPC_communication_manager_pb2.MsgValue()

        if isinstance(value, list) or isinstance(value, tuple):
            msg_value.list_msg.MergeFrom(self.create_by_type(value))
        elif isinstance(value, dict):
            if isinstance(list(value.keys())[0], str):
                msg_value.dict_msg_string_key.MergeFrom(
                    self.create_by_type(value))
            else:
                msg_value.dict_msg_int_key.MergeFrom(self.create_by_type(value))
        else:
            msg_value.single_msg.MergeFrom(self.create_by_type(value))

        return msg_value

    def transform(self, to_list=False):
        if to_list:
            self.content = self.transform_to_list(self.content)

        split_message = gRPC_communication_manager_pb2.MessageRequest()  # map/dict
        split_message.msg['message_type'].MergeFrom(
            self.build_msg_value(self.message_type))
        split_message.msg['sender'].MergeFrom(self.build_msg_value(self.sender))
        split_message.msg['receiver'].MergeFrom(
            self.build_msg_value(self.receiver))
        split_message.msg['content'].MergeFrom(self.build_msg_value(
            self.content))
        split_message.msg['communication_round'].MergeFrom(self.build_msg_value(self.communication_round))
        split_message.msg['timestamp'].MergeFrom(
            self.build_msg_value(self.timestamp))
        return split_message

    def _parse_msg(self, value):
        if isinstance(value, gRPC_communication_manager_pb2.MsgValue) or \
                isinstance(value, gRPC_communication_manager_pb2.mSingle):
            return self._parse_msg(getattr(value, value.WhichOneof("type")))
        elif isinstance(value, gRPC_communication_manager_pb2.mList):
            return [self._parse_msg(each) for each in value.list_value]
        elif isinstance(value, gRPC_communication_manager_pb2.mDict_keyIsString) or \
                isinstance(value, gRPC_communication_manager_pb2.mDict_keyIsInt):
            return {
                k: self._parse_msg(value.dict_value[k])
                for k in value.dict_value
            }
        else:
            return value

    def _parse_model(self, value):
        if isinstance(value, dict):
            return {
                k: self._parse_model(value[k]) for k in value.keys()
            }
        else:
            return pickle.loads(base64.b64decode(value))

    def parse(self, received_msg):
        self.message_type = self._parse_msg(received_msg['message_type'])
        self.sender = self._parse_msg(received_msg['sender'])
        self.receiver = self._parse_msg(received_msg['receiver'])
        self.communication_round = self._parse_msg(received_msg['communication_round'])
        self.content = self._parse_msg(received_msg['content'])
        if isinstance(self.content, dict) and "model" in self.content.keys():
            self.content["model"] = self._parse_model(self.content["model"])
        self.timestamp = self._parse_msg(received_msg['timestamp'])

    def count_bytes(self):
        """
            calculate the message bytes to be sent/received
        :return: tuple of bytes of the message to be sent and received
        """
        download_bytes = asizeof.asizeof(self.content)
        upload_cnt = len(self.receiver) if isinstance(self.receiver,
                                                      list) else 1
        upload_bytes = download_bytes * upload_cnt
        return download_bytes, upload_bytes


# if __name__ == "__main__":
#
#     msg = Message(
#         message_type=100,
#         sender="0",
#         receiver="1",
#         content={
#
#         },
#         communication_round=0
#     )
#     t = msg.transform(to_list=True)
#     print(t)
#     m = Message()
#     m.parse(t.msg)
#     print(m.message_type)

