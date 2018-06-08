# Copyright (c) 2016, hzlvtangjie. All rights reserved.

import sys, time

from .Config import Config
from .NetworkVP import NetworkVP, DqnNetworks
from .Experience import Experience
from ...game import Game
from ...game.MessageParser import GameMessageParser
from ...config import config as PkgConfig

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

        self.device = Config.DEVICE
        self.model = None

        self.master = master
        self.model_queue = Queue(maxsize=1)
        self.local_episode = 0
        
        if PkgConfig.GAME_PUSH_ALGORITHM:
            self.game = FakeGame(self.id)
        else:
            self.game = getattr(Game, PkgConfig.GAME_NAME)()


    def init_model(self, state_space_size, action_space_size):
        self.num_actions = action_space_size
        self.actions = np.arange(self.num_actions)

        model = NetworkVP(self.device, Config.NETWORK_NAME, state_space_size, action_space_size)
        target_model = NetworkVP(self.device, Config.NETWORK_NAME, state_space_size, action_space_size)
        # package model and target
        self.model = DqnNetworks(model, target_model)


    def select_action(self, state, is_test=False):
        if Config.PLAY_MODE or is_test:
            action = self.model.predict([state,])[0]
        else:
            if np.random.random() < max(0.1, 1.0-self.local_episode/20000.*0.1):
                action = np.random.randint(0, self.num_actions)
            else:
                action = self.model.predict([state,])[0]
        return action

    def run_episode(self, test=False):
        self.game.reset()
        done = False
        experiences = []

        time_count = 0
        reward_sum = 0.0

        while not done:
            # choose action
            action = self.select_action(self.game.get_state(), is_test=test)
            state, reward, done, next_state = self.game.step(action)
            reward /= PkgConfig.REWARD_SCALE
            # state, action, reward, done, next_state
            reward_sum += reward
            exp = Experience(state, action, reward, done, next_state)
            experiences.append(exp)

            if done or time_count >= Config.WORKER_DATA_SIZE:
                yield experiences, reward_sum
                # reset the tmax count
                time_count = 0
                # keep the last experience for the next batch
                experiences = []
                reward_sum = 0.0

            time_count += 1

    def run(self):
        print('Worker %d Start Running'%self.id)
        if PkgConfig.GAME_PUSH_ALGORITHM:
            self.game.run()

        state_space_size, action_space_size = self.game.get_game_model_info()
        self.init_model(state_space_size, action_space_size)
        self.master.init_queue.put((self.id, state_space_size, action_space_size))

        # init model and target model
        is_target, model = self.model_queue.get()
        self.model.update(model)
        self.model.update(model, is_target=True)

        time.sleep(np.random.rand())
        np.random.seed(np.int32(time.time() % 1 * 1000 + self.id * 10))
        while True:
            total_reward = 0
            total_length = 0
            for exps, reward_sum in self.run_episode():
                total_reward += reward_sum
                total_length += len(exps)
                # calc priority 
                self.model.calc_priority(exps)
                # put experience into training_queue
                self.master.training_queue.put(exps)

            self.local_episode += 1
            #print('local_episode', self.local_episode)
            # send log to master
            total_reward *= PkgConfig.REWARD_SCALE
            self.master.log_queue.put((total_reward, total_length))
            # every 100 episode test one game, send result to the stats
            if self.local_episode % Config.TEST_STEP == 0:
                total_reward = 0
                for exps, reward_sum in self.run_episode(test=True):
                    total_reward += reward_sum
                total_reward *= PkgConfig.REWARD_SCALE
                self.master.test_result_queue.put(total_reward)
            # update model every N episodes
            N = 50
            if self.local_episode % N != 0:
                continue
            # try to recv the model
            if not self.model_queue.empty():
                # update model
                is_target, model = self.model_queue.get()
                if is_target:
                    # update model and target
                    self.model.update(model, is_target=True)
                    self.model.update(model)
                else:
                    self.model.update(model)
