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
            while batch_size <= Config.TRAINING_MIN_BATCH_SIZE:
                id, x_, r_, a_ = self.master.training_queue.get()
                #print('get training_q', x_, r_, a_)
                if batch_size == 0:
                    x__ = x_; r__ = r_; a__ = a_
                else:
                    x__ = np.concatenate((x__, x_))
                    r__ = np.concatenate((r__, r_))
                    a__ = np.concatenate((a__, a_))
                batch_size += x_.shape[0]

            self.master.model.train(x__, r__, a__, self.id)
