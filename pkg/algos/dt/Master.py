# Copyright (c) 2016, hzlvtangjie. All rights reserved.

from threading import Thread
import socket

from .Config import Config
from .Worker import Worker
from ...game.MessageParser import MessageParser
from ...config import config as PkgConfig
from ...game import Game

from multiprocessing import Queue, Value
import time
import numpy as np


class GameListenerThread(Thread):
    """Game Listener
    """

    def __init__(self, master):
        super(GameListenerThread, self).__init__()
        self.master = master
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((PkgConfig.MASTER_IP, PkgConfig.MASTER_PORT))
        self.server.listen(32)


    def run(self):
        print("GameListenerThread starts running")
        while True:
            sock, addr = self.server.accept()
            print('Master accept connection', sock, addr)
            worker = self.master.add_worker()
            MessageParser().send(sock, worker.id)
            time.sleep(0.1)
            
class Master(object):

    MAX_RECENT_RESULTS_SIZE = 10

    def __init__(self):
        self.workers = []

        self.device = Config.MASTER_DEVICE

        self.log_queue = Queue(maxsize=100)
        self.test_flag = Value('i', 0)
        self.q_value_queue = Queue(maxsize=100)

        self.init_queue = Queue(maxsize=1)
        self.worker_cnt = 0

        self.game = getattr(Game, PkgConfig.GAME_NAME)()

        self.recent_rewards = []

    def add_worker(self):
        worker = Worker(len(self.workers), self)
        self.workers.append(worker)
        self.workers[-1].start()
        return worker

    def remove_workers(self):
        while len(self.workers) > 0:
            worker_thread = self.workers.pop(0)
            worker_thread.join()

    def run(self, init_workers):
        self.worker_cnt = init_workers

        if PkgConfig.GAME_PUSH_ALGORITHM:
            self.game_listener = GameListenerThread(self)
            self.game_listener.start()
        else:
            self.game_listener = None


        if not PkgConfig.GAME_PUSH_ALGORITHM:
            for _ in range(init_workers):
                self.add_worker()

        f = open('results/dt_result.txt', 'w')
        while True:
            #start test
            total_reward = 0
            self.game.reset()
            done = False
            while not done:
                state = self.game.get_state()
                for worker in self.workers:
                    worker.state_queue.put(state)
                # compute average q_values from all trees
                action_cnts = {}
                for _ in range(len(self.workers)):
                    episode_count, q_values = self.q_value_queue.get()
                    w_action = np.argmax(q_values)
                    action_cnts[w_action] = action_cnts.get(w_action, 0) + 1
                sorted_cnts = sorted(action_cnts.values(), reverse=True)
                action = [key for key, value in action_cnts.items() if value == sorted_cnts[0]][0]
                state, reward, done, next_state = self.game.step(action)
                total_reward += reward
                self.recent_rewards.append(total_reward)
                while len(self.recent_rewards) > self.MAX_RECENT_RESULTS_SIZE:
                    self.recent_rewards.pop(0)
                avg_reward = sum(self.recent_rewards)/float(len(self.recent_rewards))
            print(
                'Episode %d: Reward %.2f'
                %(episode_count, avg_reward)
                )
            f.write('%d %.2f\n'%(episode_count, avg_reward))
            f.flush()
            # send done flag to workers
            for worker in self.workers:
                worker.state_queue.put(True)
        f.close()

        # close game_listener
        if self.game_listener:
            self.game_listener.join()
        # remove worker thread
        self.remove_workers()
