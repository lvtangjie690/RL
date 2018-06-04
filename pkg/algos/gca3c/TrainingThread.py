from threading import Thread
import numpy as np
from .Config import Config

class TrainingThread(Thread):
    """Master's training thread
    """

    def __init__(self, master, id):
        super(TrainingThread, self).__init__()
        self.master = master 
        self.id = id

    def run(self):
        print("TrainingThread starts running")
        while True:
            batch_size = 0
            worker_ids = {}
            while batch_size <= Config.TRAINING_MIN_BATCH_SIZE:
                id, x_, r_, a_ = self.master.training_queue.get()
                if Config.USE_EXP_CACHE:
                    self.master.put_sample(x_, r_, a_)
                    x_, r_, a_ = self.master.get_sample()
                worker_ids[id] = 1
                #print('get training_q', x_, r_, a_)
                if batch_size == 0:
                    x__ = x_; r__ = r_; a__ = a_
                else:
                    x__ = np.concatenate((x__, x_))
                    r__ = np.concatenate((r__, r_))
                    a__ = np.concatenate((a__, a_))
                batch_size += x_.shape[0]

            self.master.model.train(x__, r__, a__, self.id)
            # send new model to workers
            new_model = self.master.model.dumps()
            for worker_id in worker_ids.keys():
                self.master.workers[worker_id].model_queue.put(new_model)
