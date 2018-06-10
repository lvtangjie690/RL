from threading import Thread

class TrainingThread(Thread):
    """Master's training thread
    """

    def __init__(self, master, id):
        super(TrainingThread, self).__init__()
        self.master = master 
        self.id = id

    def run(self):
        print('TrainingThread Starts Running')
        while True:
            put_back, exps = self.master.sampled_queue.get()
            self.master.model.train(exps)
            if put_back:
                self.master.model.calc_priority(exps)
                self.master.training_queue.put(exps)
            with self.master.training_lock:
                #print('!!!', self.master.model.calc_q_labels(exps))
                self.master.training_step += 1
                if self.master.training_step % 1000 == 0:
                    self.master.model.replace_target()
                    target = self.master.model.dumps(is_target=True)
                    for worker in self.master.workers:
                        worker.model_queue.put((True, target))
                else:
                    model = self.master.model.dumps()
                    for worker in self.master.workers:
                        if not worker.model_queue.full():
                            worker.model_queue.put((False, model))
