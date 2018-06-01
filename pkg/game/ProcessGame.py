import socket
import sys
import time

from .MessageParser import MessageParser, GameMessageParser
from ..config import config as PkgConfig
from . import Game

class ProcessGame(object):

    def __init__(self):
        self.game = getattr(Game, PkgConfig.GAME_NAME)()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.msg_parser = GameMessageParser()


    def run(self):
        # connect master to get worker's id  
        self.sock.connect((PkgConfig.MASTER_IP, PkgConfig.MASTER_PORT))
        worker_id = MessageParser().decode_recv(self.sock)
        print('Game recv worker id', worker_id)

        self.sock.shutdown(2)
        self.sock.close()

        # try to connect worker
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        while True:
            try:
                self.sock.connect(('localhost', PkgConfig.WORKER_BASE_PORT+worker_id))
                break
            except Exception as e:
                print(e)
                time.sleep(1)
        

        self.msg_parser.send_game_model_info(self.sock, self.game.get_game_model_info())

        state, reward, done, next_state = self.game.reset()
        while True:
            self.msg_parser.send_sample(self.sock, [state, reward, done, next_state])
            if not done:
                action = self.msg_parser.recv_action(self.sock)
                state, reward, done, next_state = self.game.step(action)
            else:
                state, reward, done, next_state = self.game.reset()


if __name__ == '__main__':
    game = ProcessGame()
    game.run()
