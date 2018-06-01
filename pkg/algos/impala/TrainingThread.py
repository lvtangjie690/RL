from threading import Thread
from .Config import Config
import numpy as np
import time

class TrainingThread(Thread):
    """Master's training thread
    """

    def __init__(self, master, id):
        super(TrainingThread, self).__init__()
        self.master = master 
        self.id = id

    def handle_exps(self, exps, terminal_v):
        if len(exps) <= 0:
            return None
        if Config.OFF_POLICY_CORRECTION:
            exp_states = [exp.state for exp in exps]
            pis, _ = self.master.model.predict_p_and_v(exp_states)
            ## compute rho
            ratios = [pis[id][exps[id].action] / exps[id].miu for id in range(len(exps))]

            rho_ = [min(Config.RHO, rho) for rho in ratios]
            c_ = [min(Config.C, rho) for rho in ratios]

            delta_vs = []
            for idx, exp in enumerate(exps):
                if idx == len(exps)-1:
                    delta_vs.append(rho_[idx]*(exp.reward + Config.DISCOUNT * terminal_v - exp.value))
                else:
                    delta_vs.append(rho_[idx]*(exp.reward + Config.DISCOUNT * exps[idx+1].value - exp.value))

            second_sum = 0.
            for idx in reversed(range(0, len(exps))):
                second_sum += delta_vs[idx] 
                exps[idx].second_sum = second_sum
                # second sum
                c_i = 1 if idx <= 0 else c_[idx-1]
                second_sum *= (Config.DISCOUNT * c_i)
            ## 
            for idx in reversed(range(0, len(exps))):
                exps[idx].v = exps[idx].value + exps[idx].second_sum
                if idx == len(exps)-1:
                    next_v = terminal_v
                else:
                    next_v = exps[idx+1].v
                exps[idx].reward_sum = exps[idx].reward + Config.DISCOUNT * next_v
                exps[idx].rho = rho_[idx]
        else:
            for exp in exps:
                exp.v = exp.reward_sum
                exp.rho = 1.

        x = np.array([exp.state for exp in exps])
        a = np.eye(self.master.model.num_actions)[np.array([exp.action for exp in exps])].astype(np.float32)
        y_r = np.array([exp.reward_sum for exp in exps])
        v = np.array([exp.v for exp in exps])
        #v = np.array([exp.reward_sum for exp in exps])
        rho = np.array([exp.rho for exp in exps])
        #rho = np.array([1. for exp in exps])

        return x, a, y_r, v, rho

    def do_train(self, from_cache=False):
        batch_size = 0
        worker_ids = {}
        while batch_size <= Config.TRAINING_MIN_BATCH_SIZE:
            if from_cache:
                exps, terminal_v = self.master.get_sample()   
            else:
                id, exps, terminal_v = self.master.training_queue.get()
                worker_ids[id] = 1
                self.master.put_sample(exps, terminal_v)
                exps, terminal_v = self.master.get_sample()   
            #
            x_, a_, r_, v_, rho_ = self.handle_exps(exps, terminal_v)
            if batch_size == 0:
                x__ = x_; r__ = r_; a__ = a_
                v__ = v_; rho__ = rho_
            else:
                x__ = np.concatenate((x__, x_))
                r__ = np.concatenate((r__, r_))
                a__ = np.concatenate((a__, a_))
                v__ = np.concatenate((v__, v_))
                rho__ = np.concatenate((rho__, rho_))
            batch_size += x_.shape[0]

        self.master.model.train(x__, a__, r__, v__, rho__, self.id)
        return worker_ids


    def run(self):
        print("TrainingThread starts running")
        while True:
            # on-policy train with off-policy correction
            worker_ids = self.do_train()
            #for _ in range(2):
            #    # off-policy train
            #    self.do_train(from_cache=True)
            new_model = self.master.model.dumps()
            ## send new model to workers
            for worker_id in worker_ids.keys():
                worker = self.master.workers[worker_id]
                if not worker.model_queue.full():
                    worker.model_queue.put(new_model)
