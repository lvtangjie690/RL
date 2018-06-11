# Copyright (c) 2016, hzlvtangjie. All rights reserved.

from threading import Thread, Lock
import socket, time
from multiprocessing import Queue

from .Config import Config
from .NetworkVP import NetworkVP, DqnNetworks
from .Worker import Worker
from ...game.MessageParser import MessageParser
from ...config import config as PkgConfig
from .ReplayBuffer import ReplayBuffer, PrioritizedReplayBuffer
from .TrainingThread import TrainingThread
from .Stats import Stats


class InitListenerThread(Thread):
    """init model listener
    """
    
    def __init__(self, master):
        super(InitListenerThread, self).__init__()
        self.master = master

    def run(self):
        print("InitListenerThread starts running")
        while True:
            id, state_space_size, action_space_size = self.master.init_queue.get()
            if self.master.model is None:
                self.master.init_model(state_space_size, action_space_size)
            self.master.workers[id].model_queue.put((False, self.master.model.dumps()))

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

    def __init__(self):
        self.workers = []
        self.trainers = []
        self.replay_buffers = []

        self.device = Config.MASTER_DEVICE
        self.model = None

        self.log_queue = Queue(maxsize=100)
        self.test_result_queue = Queue(maxsize=100)
        # 
        self.stats = Stats(self.log_queue, self.test_result_queue)

        self.training_queue = Queue(maxsize=4096)
        self.sampled_queue = Queue(maxsize=128)
        self.init_queue = Queue(maxsize=1)

        self.training_step = 0
        self.training_lock = Lock()

    def init_model(self, state_space_size, action_space_size):
        model = NetworkVP(self.device, Config.NETWORK_NAME, state_space_size, action_space_size)
        if Config.LOAD_CHECKPOINT:
            model.load()
        target_model = NetworkVP(self.device, Config.NETWORK_NAME, state_space_size, action_space_size)
        target_model.update(model.dumps())
        self.model = DqnNetworks(model, target_model)

    def add_worker(self):
        worker = Worker(len(self.workers), self)
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

    def add_replay_buffer(self):
        if Config.USE_PRIORITY:
            rb = PrioritizedReplayBuffer(self.training_queue, self.sampled_queue)
        else:
            rb = ReplayBuffer(self.training_queue, self.sampled_queue)
        self.replay_buffers.append(rb)
        self.replay_buffers[-1].start()

    def remove_replay_buffers(self):
        while len(self.replay_buffers) > 0:
            replay_buffer = self.replay_buffers.pop(0)
            replay_buffer.join()


    def run(self, init_workers):
        self.stats.start()

        self.init_listener = InitListenerThread(self)
        self.init_listener.start()

        if PkgConfig.GAME_PUSH_ALGORITHM:
            self.game_listener = GameListenerThread(self)
            self.game_listener.start()
        else:
            self.game_listener = None

        if not PkgConfig.GAME_PUSH_ALGORITHM:
            for _ in range(init_workers):
                self.add_worker()

        for _ in range(Config.BUFFERS):
            self.add_replay_buffer()

        for _ in range(Config.TRAINERS):
            self.add_trainer()


        while True:
            if self.stats.save_flag.value:
                self.model.save(self.stats.episode_count.value)
                self.stats.save_flag.value = 0
            time.sleep(1)

        # close init_listener
        self.init_listener.join()
        # close game_listener
        if self.game_listener:
            self.game_listener.join()
        # remove worker thread
        self.remove_workers()
        # remove trainer thread
        self.remove_trainers()
        # close stats process 
        self.stats.join()
        # remove replay buffers
        self.remove_replay_buffers()
