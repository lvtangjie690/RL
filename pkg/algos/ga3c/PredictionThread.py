from threading import Thread
from .Config import Config

class PredictionThread(Thread):
    """Master's prediction thread
    """

    def __init__(self, master, id):
        super(PredictionThread, self).__init__()
        self.master = master
        self.id = id 

    def run(self):
        print("PredictionThread starts running")
        while True:
            worker_ids = []
            states = []
            id, state = self.master.prediction_queue.get()
            worker_ids.append(id)
            states.append(state)
            batch_size = 1
            # wait for other worker's state
            while not self.master.prediction_queue.empty() and batch_size \
                    < Config.PREDICTION_MAX_BATCH_SIZE:
                id, state = self.master.prediction_queue.get()
                worker_ids.append(id)
                batch_size += 1
                states.append(state)

            predictions, values = self.master.model.predict_p_and_v(states)
            for idx, worker_id in enumerate(worker_ids):
                self.master.workers[worker_id].result_queue.put((predictions[idx], values[idx]))
                
