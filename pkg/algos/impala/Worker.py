# Copyright (c) 2016, hzlvtangjie. All rights reserved.

import sys, time

from .Config import Config
from .NetworkVP import NetworkVP
from .Experience import Experience
from ...game import Game
from ...game.MessageParser import GameMessageParser
from ...config import config as PkgConfig

import numpy as np
import queue
import socket
from multiprocessing import Process, Queue
import random

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
        self.model_queue = Queue(maxsize=100)
        
        if PkgConfig.GAME_PUSH_ALGORITHM:
            self.game = FakeGame(self.id)
        else:
            self.game = getattr(Game, PkgConfig.GAME_NAME)()


    def init_model(self, state_space_size, action_space_size):
        self.num_actions = action_space_size
        self.actions = np.arange(self.num_actions)

        self.model = NetworkVP(self.device, Config.NETWORK_NAME, state_space_size, action_space_size)

    def select_action(self, prediction, is_test=False):
        if Config.PLAY_MODE or is_test:
            action = np.argmax(prediction)
        else:
            action = np.random.choice(self.actions, p=prediction)
        return action

    def predict_p_and_v(self, state):
        predictions, values = self.model.predict_p_and_v([state,])
        return predictions[0], values[0]

    def run_episode(self, test=False):
        self.game.reset()
        done = False
        experiences = []

        time_count = 0
        reward_sum = 0.0

        while not done:
            prediction, value = self.predict_p_and_v(self.game.get_state())
            action = self.select_action(prediction, is_test=test)
            state, reward, done, next_state = self.game.step(action)
            reward /= PkgConfig.REWARD_SCALE
            # state, action, reward, done, next_state
            reward_sum += reward
            exp = Experience(state, action, prediction, reward, done, value)
            experiences.append(exp)

            if done or time_count == PkgConfig.TIME_MAX:
                #exp_length = len(experiences) if done else len(experiences) - 1
                exp_length = len(experiences)
                terminal_v = 0 if done else experiences[-1].value
                r_ = terminal_v
                iteration_range = exp_length if done else exp_length-1
                for t in reversed(range(0, iteration_range)):
                    r = np.clip(experiences[t].reward, PkgConfig.REWARD_MIN, PkgConfig.REWARD_MAX)
                    r_ = Config.DISCOUNT * r_ + r
                    experiences[t].reward_sum = r_

                updated_exps = experiences[:exp_length]
                
                yield updated_exps, reward_sum

                # reset the tmax count
                time_count = 0
                # keep the last experience for the next batch
                experiences = [experiences[-1]]
                reward_sum = 0.0

            time_count += 1

    def run(self):
        print('Worker %d Start Running'%self.id)
        if PkgConfig.GAME_PUSH_ALGORITHM:
            self.game.run()

        state_space_size, action_space_size = self.game.get_game_model_info()
        self.init_model(state_space_size, action_space_size)

        self.master.init_queue.put((self.id, state_space_size, action_space_size))
        model = self.model_queue.get()
        self.model.update(model)

        time.sleep(np.random.rand())
        np.random.seed(np.int32(time.time() % 1 * 1000 + self.id * 10))

        self.local_episode = 0
        while True:
            total_reward = 0
            total_length = 0
            for exps, reward_sum in self.run_episode():
                total_reward += reward_sum
                total_length += len(exps)
                # send training data to the master
                self.master.training_queue.put((self.id, exps))
                # recv model from master
                model = None
                while not self.model_queue.empty():
                    model = self.model_queue.get()
                if model:
                    self.model.update(model)
                    #print('Worker %d update'%self.id)

            # send log to master
            total_reward *= PkgConfig.REWARD_SCALE
            self.master.log_queue.put((total_reward, total_length))
            # test game
            self.local_episode += 1
            if self.local_episode % Config.TEST_STEP == 0:
                total_reward = 0
                for exps, reward_sum in self.run_episode(test=True):
                    total_reward += reward_sum
                total_reward *= PkgConfig.REWARD_SCALE
                self.master.result_queue.put(total_reward)
