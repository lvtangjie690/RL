import struct
import platform
import json
import numpy as np

USING_PYTHON3 = True if \
    platform.python_version()[0] == '3' else False

MAX_RECV_SIZE = 2048

class MessageParser(object):

    def __init__(self):
        if USING_PYTHON3:
            self.last_msg = b''
        else:
            self.last_msg = ''

    def recv(self, sock):
        while len(self.last_msg) <= 4:
            self.last_msg += sock.recv(MAX_RECV_SIZE)
        msg_length = struct.unpack('I', self.last_msg[:4])[0]
        self.last_msg = self.last_msg[4:]
        while len(self.last_msg) < msg_length:
            self.last_msg += sock.recv(MAX_RECV_SIZE)
        msg = self.last_msg[:msg_length]
        if USING_PYTHON3:
            msg = msg.decode('utf-8')
        self.last_msg = self.last_msg[msg_length:]
        return msg

    def send(self, sock, data):
        json_msg = json.dumps(data)
        if USING_PYTHON3:
            json_msg = json_msg.encode('utf-8')
        msg_length = len(json_msg)
        length_msg = struct.pack('I', msg_length)
        sock.sendall(length_msg+json_msg)

    def decode_recv(self, sock):
        msg = self.recv(sock)
        return json.loads(msg)

class GameMessageParser(MessageParser):

    def send_sample(self, sock, sample):
        self.send(sock, sample)

    def recv_action(self, sock):
        action = json.loads(self.recv(sock))
        return action

    def send_action(self, sock ,action):
        action = np.asscalar(action)
        self.send(sock, action)
    
    def recv_sample(self, sock):
        sample = json.loads(self.recv(sock))
        return sample

    def send_game_model_info(self, sock, info):
        """send state_size and action_size for model init
        """
        self.send(sock, info)

    def recv_game_model_info(self, sock):
        return json.loads(self.recv(sock))
