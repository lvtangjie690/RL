import bisect
from multiprocessing import Queue, Process
import numpy as np
from .Config import Config
import time, random

class ReplayBuffer(Process):

    def __init__(self, training_queue, sampled_queue):
        super(ReplayBuffer, self).__init__()
        self.training_queue = training_queue
        self.sampled_queue = sampled_queue
        # experience_replay 
        self.buffer = []
        self.max_buffer_size = Config.MAX_BUFFER_SIZE

    def add(self, exps):
        """add replay buffer
        """
        self.buffer.extend(exps)
        while len(self.buffer) > self.max_buffer_size:
            self.buffer.pop(-1)

    def sample(self):
        if len(self.buffer) < Config.TRAINING_BATCH_SIZE:
            return False, None
        return False, random.sample(self.buffer, Config.TRAINING_BATCH_SIZE)
       
    def run(self):
        print('ReplayBuffer Starts Running')
        while True:
            # if not empty, get new data
            if not self.training_queue.empty():
                exps = self.training_queue.get()
                self.add(exps)
            # sample data and send to the master update
            if not self.sampled_queue.full():
                put_back, exps = self.sample()
                if exps != None:
                    self.sampled_queue.put((put_back, exps))

class PrioritizedReplayBuffer(ReplayBuffer):

    def __init__(self, training_queue, sampled_queue):
        super(PrioritizedReplayBuffer, self).__init__(training_queue, sampled_queue)
        # prepare for sample and update
        self.s_list = []
        self.w_list = []
        self.last_segments_size = 0


    def calc_segments_and_weights(self):
        #print('calc_segments_and_weights', len(self.buffer))
        size = len(self.buffer)
        alpha = Config.P_ALPHA
        beta = min(Config.P_MAX_BETA,  Config.P_BETA_BASE + size/Config.P_ANNEALED_STEP*0.1)
       
        # compute probability 
        p_list = [(1./i)**alpha for i in range(size, 0, -1)]
        total_p = sum(p_list)
        p_list = np.array([1/total_p for i in p_list])
        # compute weights        
        w_list = (p_list*size)**(-1.*beta)
        max_weight = np.max(w_list)
        if max_weight > 0:
            w_list = w_list / max_weight
        # compute segments
        avg_p = 1./Config.TRAINING_BATCH_SIZE
        s_list = []
        now_total_p = 0.
        for idx, p in enumerate(p_list):
            now_total_p += p
            if now_total_p >= avg_p:
                s_list.append(idx)
                now_total_p -= avg_p
            if len(s_list) >= Config.TRAINING_BATCH_SIZE:
                break
        # no enough segments
        while len(s_list) < Config.TRAINING_BATCH_SIZE:
            s_list.append(size-1)

        self.s_list = s_list
        self.w_list = w_list
        # record last segments size
        self.last_segments_size = len(self.buffer) - len(self.buffer)%Config.MIN_BUFFER_SIZE


    # put experiences in order
    def add(self, exps):
        for exp in exps:
            bisect.insort_left(self.buffer, exp)
        while len(self.buffer) > self.max_buffer_size:
            self.buffer.pop(-1)
         
        if len(self.buffer) >= (self.last_segments_size+Config.MIN_BUFFER_SIZE):
            self.calc_segments_and_weights()
   
    # sample experiences from buffer 
    def sample(self):
        if len(self.buffer) < Config.TRAINING_BATCH_SIZE:
            return False, None
        # no rank list
        if len(self.s_list) <= 0:
            exps = random.sample(self.buffer, Config.TRAINING_BATCH_SIZE)
            return False, exps
        sampled_idx_list = []
        exps = []
        # select sampled data's idx
        for i in range(Config.TRAINING_BATCH_SIZE):
            if i == 0: 
                start, end = 0, self.s_list[i]+1
            else:
                start, end = self.s_list[i-1]+1, self.s_list[i]+1
            if start >= end:
                continue
            sampled_idx = np.random.randint(start, end)
            sampled_idx_list.append(sampled_idx)
        # do pop exps
        exps = []
        offset = 0
        for sampled_idx in sampled_idx_list:
            true_idx = sampled_idx - offset
            if true_idx >= len(self.buffer):
                break
            exp = self.buffer.pop(true_idx)
            if sampled_idx >= len(self.w_list):
                exp.weight = 0
            else:
                exp.weight = self.w_list[sampled_idx]
            exps.append(exp)
            offset += 1
        return True, exps
