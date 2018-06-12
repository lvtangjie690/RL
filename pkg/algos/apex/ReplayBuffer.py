import bisect
from multiprocessing import Process, Queue
import numpy as np
from .Config import Config
import random

class ReplayBuffer(Process):

    def __init__(self, id, training_queue, sampled_queue):
        super(ReplayBuffer, self).__init__()
        self.training_queue = training_queue
        self.sampled_queue = sampled_queue
        #
        self.unique_training_queue = Queue(maxsize=128)
        # experience_replay 
        self.buffer = []
        self.max_buffer_size = Config.MAX_BUFFER_SIZE
        # replay_buffer id
        self.id = id

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
            if not self.unique_training_queue.empty():
                exps = self.unique_training_queue.get()
                self.add(exps)
            # sample data and send to the master update
            if not self.sampled_queue.full():
                put_back, exps = self.sample()
                if exps != None:
                    self.sampled_queue.put((self.id, put_back, exps))

class RPReplayBuffer(ReplayBuffer):
    """rank based prioritized replay buffer
    """

    def __init__(self, id, training_queue, sampled_queue):
        super(RPReplayBuffer, self).__init__(id, \
            training_queue, sampled_queue)
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
        return False, exps

class SumTree(object):
    """refer to openai baselines(SegmentTree)
    """

    def __init__(self, capacity):
        
        self._capacity = capacity
        self._value = [0 for _ in range(2*capacity)]

        self._min_val = 1e10


    def __setitem__(self, idx, val):
        idx += self._capacity
        self._value[idx] = val

        if val < self._min_val:
            self._min_val = val
        
        idx = int(idx/2)
        while idx >= 1:
            self._value[idx] = self._value[2*idx] + self._value[2*idx+1]
            idx  = int(idx/2)


    def __getitem__(self, idx):
        return self._value[idx+self._capacity]

    
    def sum(self):
        return self._value[1]

    def min(self):
        return self._min_val

    def sample_idx(self, priority):
        idx = 1
        while idx < self._capacity:
            if self._value[2*idx] > priority:
                idx *= 2
            else:
                priority -= self._value[2*idx]
                idx = 2 * idx + 1

        return idx - self._capacity

class PPReplayBuffer(ReplayBuffer):
    """ proportional prioriized replay buffer
    """

    def __init__(self, id, training_queue, sampled_queue):
        super(PPReplayBuffer, self).__init__(id, \
            training_queue, sampled_queue)
        capacity = 1
        while capacity < Config.MAX_BUFFER_SIZE:
            capacity *= 2
        # should be power of 2
        self.max_buffer_size = capacity
        self.tree = SumTree(capacity)
        # 
        self.now_idx = 0
        self.buffer = [None for _ in range(self.max_buffer_size)]
        #
        self.step = 0
        # now_size
        self.N = 0

    def add_exp(self, exp):
        self.step += 1
        if exp.id != None:
            self.buffer[exp.id] = exp
            self.tree[exp.id] = exp.priority ** Config.P_ALPHA
        else:
            exp.id = self.now_idx
            self.buffer[exp.id] = exp
            self.tree[exp.id] = exp.priority ** Config.P_ALPHA
            self.now_idx = (self.now_idx + 1) % self.max_buffer_size
            if self.N < self.max_buffer_size:
                self.N += 1

    def add(self, exps):
        """add replay buffer
        """
        for exp in exps:
            self.add_exp(exp)

    def sample(self):
        if self.N < Config.TRAINING_BATCH_SIZE:
            return False, None

        beta = min(Config.P_MAX_BETA,  Config.P_BETA_BASE + self.step/Config.P_ANNEALED_STEP*0.1)

        total_priority = self.tree.sum() + Config.PRIORITY_EPSILON
        max_weight = (self.tree.min() / total_priority * self.N) ** (-beta)
        exps = []
        for _ in range(Config.TRAINING_BATCH_SIZE):
            priority = np.random.random() * total_priority
            idx = self.tree.sample_idx(priority)
            exp = self.buffer[idx]
            prob = self.tree[idx] / total_priority
            exp.weight = (prob * self.N) **(-beta) / max_weight
            exps.append(exp)
        return True, exps
