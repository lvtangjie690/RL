# Copyright (c) 2016, hzlvtangjie. All rights reserved.

from threading import Thread
import socket

from .Config import Config
from .NetworkVP import NetworkVP
from .TrainingThread import TrainingThread
from .PredictionThread import PredictionThread
from .Worker import Worker
from ...game.MessageParser import MessageParser
from ...config import config as PkgConfig

from multiprocessing import Queue, Process, Value, Lock
import time

init_model_lock = Lock()

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
            

class Stats(Process):

    MAX_RECENT_RESULTS_SIZE = 10

    def __init__(self, log_queue, result_queue):
        super(Stats, self).__init__()
        self.log_queue = log_queue
        self.result_queue = result_queue
       
        self.episode_count = 0 
        self.recent_reward_results = []
        self.frame_count = 0

        self.last_frame_count = 0
        self.last_time = 0
        self.last_fps = 0

    def run(self):
        print("Stats starts running")
        self.last_time = time.time()
        with open('results/ga3c_result.txt', 'w') as f:
            while True:
                total_reward, total_length = self.log_queue.get()
                self.episode_count += 1
                self.frame_count += total_length
                while not self.result_queue.empty():
                    test_reward = self.result_queue.get()
                    self.recent_reward_results.append(test_reward)
                while len(self.recent_reward_results) > self.MAX_RECENT_RESULTS_SIZE:
                    self.recent_reward_results.pop(0)
                recent_avg_reward = sum(self.recent_reward_results)/float(len(self.recent_reward_results)) \
                    if len(self.recent_reward_results) > 0 else None

                if self.episode_count % 100 == 0:
                    if time.time() - self.last_time > 10:
                        self.last_fps = int((self.frame_count - self.last_frame_count)/(time.time() - self.last_time))
                        self.last_frame_count = self.frame_count
                        self.last_time = time.time()
                    elif self.last_fps == 0:
                        self.last_fps = int(self.frame_count/(time.time() - self.last_time))
                    if recent_avg_reward is None:
                        continue
                    print(
                        'Episode %d: Recent Avg Reward %.2f, FPS %d'\
                        %(self.episode_count, recent_avg_reward, self.last_fps)
                        )
                    f.write('%d %.2f\n'%(self.episode_count, recent_avg_reward))
                    f.flush()
        


class Master(object):

    def __init__(self):
        self.workers = []
        self.trainers = []
        self.predictors = []

        self.device = Config.MASTER_DEVICE
        self.model = None
        self.training_queue = Queue(maxsize=2048)
        self.prediction_queue = Queue(maxsize=2048)

        self.log_queue = Queue(maxsize=100)
        self.result_queue = Queue(maxsize=100)
        self.stats = Stats(self.log_queue, self.result_queue)

        self.init_queue = Queue(maxsize=1)
        self.state_space_size = Value('i', 0)
        self.action_space_size = Value('i', 0)


    def init_model(self, state_space_size, action_space_size):
        self.model = NetworkVP(self.device, Config.NETWORK_NAME, state_space_size, action_space_size)

    def add_worker(self):
        worker = Worker(len(self.workers), self, init_model_lock)
        self.workers.append(worker)
        self.workers[-1].start()
        return worker

    def remove_workers(self):
        while len(self.workers) > 0:
            worker_thread = self.workers.pop(0)
            worker_thread.join()

    def add_trainer(self):
        trainer = TrainingThread(self, len(self.trainers))
        self.trainers.append(trainer)
        self.trainers[-1].start()

    def remove_trainers(self):
        while len(self.trainers) > 0:
            trainer_thread = self.trainers.pop(0)
            trainer_thread.join()

    def add_predictor(self):
        predictor = PredictionThread(self, len(self.predictors))
        self.predictors.append(predictor)
        self.predictors[-1].start()

    def remove_predictors(self):
        while len(self.predictors) > 0:
            predictor_thread = self.predictors.pop(0)
            predictor_thread.join()

    def run(self, init_workers):
        self.stats.start()

        if PkgConfig.GAME_PUSH_ALGORITHM:
            self.game_listener = GameListenerThread(self)
            self.game_listener.start()
        else:
            self.game_listener = None

        # first add workers
        if not PkgConfig.GAME_PUSH_ALGORITHM:
            for _ in range(init_workers):
                self.add_worker()

        # wait for game's state space and action space size 
        while True:
            with init_model_lock:
                if self.state_space_size.value != 0 and self.action_space_size.value != 0:
                    self.init_model(self.state_space_size.value, self.action_space_size.value)
                    break
            time.sleep(1) 

        # add trainer
        for _ in range(Config.TRAINERS):
            self.add_trainer()
        # add predictor 
        for _ in range(Config.PREDICTORS):
            self.add_predictor()


        while True:
            time.sleep(10)

        # close game_listener
        if self.game_listener:
            self.game_listener.join()
        # remove worker thread
        self.remove_workers()
        # remove trainer thread
        self.remove_trainers()
        # close stats process 
        self.stats.join()
