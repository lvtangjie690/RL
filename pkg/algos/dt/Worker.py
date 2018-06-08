# Copyright (c) 2016, hzlvtangjie. All rights reserved.

import sys, time

from .Config import Config
from .DecisionTree import DecisionTree
from ...game import Game
from ...config import config as PkgConfig
from ...game.MessageParser import GameMessageParser

import numpy as np
import queue
import socket

from multiprocessing import Process, Queue

class FakeGame(object):
    """ FakeGame: recv and send real game's msg and \
        step as a game for worker
    """

    def __init__(self, id):
        self.id = id
        self.state_queue = queue.Queue(maxsize=1)

        self.last_action = None        
        self.game_msg_parser = GameMessageParser()
        #init game listener
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        port = PkgConfig.WORKER_BASE_PORT + self.id
        self.server.bind(('localhost', port))
        self.server.listen(1)

    def run(self):
        while True:
            # create connection between worker and game
            self.sock, addr = self.server.accept()
            break

    def get_state(self):
        if self.state_queue.empty():
            # the first frame in an episode
            sample = self.game_msg_parser.recv_sample(self.sock)
            state, reward, done, next_state = sample
            return next_state
        else:
            return self.state_queue.get()            


    def step(self, action):
        # send action
        self.game_msg_parser.send_action(self.sock, action)
        # recv new sample
        sample = self.game_msg_parser.recv_sample(self.sock) 
        state, reward, done, next_state = sample
        if not done:
            self.state_queue.put(next_state)
        return sample
    
    def reset(self):
        pass

    def get_game_model_info(self):
        return self.game_msg_parser.recv_game_model_info(self.sock)

class Worker(Process):
    def __init__(self, id, master):
        super(Worker, self).__init__()
        self.id = id

        self.discount_factor = Config.DISCOUNT

        self.device = Config.WORKER_DEVICE
        self.model = None

        self.master = master
        self.state_queue = Queue(maxsize=10)
        
        if PkgConfig.GAME_PUSH_ALGORITHM:
            self.game = FakeGame(self.id)
        else:
            self.game = getattr(Game, PkgConfig.GAME_NAME)()

        self.episode_count = 0


    def init_model(self, state_space_size, action_space_size):
        self.num_actions = action_space_size
        self.actions = np.arange(self.num_actions)
        
        self.model = DecisionTree(action_space_size)

    def select_action(self, state):
        if np.random.random() < max(0.1, 1.0-self.episode_count/Config.ANNEALING_RATE):
            action = np.random.choice(self.actions)
        else:
            action = self.model.predict(state)
        return action

    def predict_p_and_v(self, state):
        predictions, values = self.model.predict_p_and_v([state,])
        return predictions[0], values[0]

    def run_episode(self):
        self.game.reset()
        done = False

        frame_count = 0
        reward_sum = 0.0
        while not done:
            # very first few frames
            action = self.select_action(self.game.get_state())
            state, reward, done, next_state = self.game.step(action)
            reward /= PkgConfig.REWARD_SCALE 
            # state, action, reward, done, next_state
            reward_sum += reward
            self.model.train(state, action, reward, done, next_state)
            frame_count += 1

        return reward_sum, frame_count

    def run(self):
        print('Worker %d Start Running'%self.id)
        if PkgConfig.GAME_PUSH_ALGORITHM:
            self.game.run()

        state_space_size, action_space_size = self.game.get_game_model_info()
        self.init_model(state_space_size, action_space_size)

        time.sleep(np.random.rand())
        np.random.seed(np.int32(time.time() % 1 * 1000 + self.id * 10))
        
        while True:
            total_reward, total_length = self.run_episode()
            total_reward *= PkgConfig.REWARD_SCALE
            self.episode_count += 1

            if self.episode_count % Config.TEST_STEP == 0:
                while True:
                    if self.state_queue.empty():
                        # wait for new state
                        time.sleep(0.1)
                    else: 
                        state = self.state_queue.get()
                        # done flag
                        if state == True:
                            break
                        q_values = self.model.get_q_values(state)
                        self.master.q_value_queue.put((self.episode_count, q_values))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python Worker.py id")
        sys.exit(0)
    worker = Worker(int(sys.argv[1]))
    worker.run()
