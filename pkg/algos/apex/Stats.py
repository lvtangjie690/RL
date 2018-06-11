from multiprocessing import Process, Value
import time
from .Config import Config

class Stats(Process):

    MAX_RECENT_RESULTS_SIZE = 50

    def __init__(self, log_queue, result_queue):
        super(Stats, self).__init__()
        self.log_queue = log_queue
        self.result_queue = result_queue
       
        self.episode_count = Value('i', 0)
        self.recent_reward_results = []
        self.frame_count = 0

        self.last_frame_count = 0
        self.last_time = 0
        self.last_fps = 0

        self.save_flag = Value('i', 0)

    def run(self):
        print("Stats starts running")
        self.last_time = time.time()
        f = open('results/apex_result.txt', 'w')
        while True:
            total_reward, total_length = self.log_queue.get()
            self.episode_count.value += 1
            self.frame_count += total_length
            if not self.result_queue.empty():
                reward = self.result_queue.get()
                self.recent_reward_results.append(reward)
                while len(self.recent_reward_results) > self.MAX_RECENT_RESULTS_SIZE:
                    self.recent_reward_results.pop(0)
            recent_avg_reward = None if len(self.recent_reward_results) <= 0 else \
                sum(self.recent_reward_results)/float(len(self.recent_reward_results))

            if self.episode_count.value % Config.STATS_SHOW_STEP == 0:
                if time.time() - self.last_time > 10:
                    self.last_fps = int((self.frame_count - self.last_frame_count)/(time.time() - self.last_time))
                    self.last_frame_count = self.frame_count
                    self.last_time = time.time()
                elif self.last_fps == 0:
                    self.last_fps = int(self.frame_count/(time.time() - self.last_time))
                if recent_avg_reward is None:
                    continue
                print(
                    'Episode %d; Reward %.2f FPS %d' \
                    %(self.episode_count.value, recent_avg_reward, self.last_fps)
                    )
                f.write('%d %.2f\n'%(self.episode_count.value, recent_avg_reward))
                f.flush()
            if Config.SAVE_FREQUENCY and self.episode_count.value % Config.SAVE_FREQUENCY == 0:
                self.save_flag.value = 1

            while self.save_flag.value:
                time.sleep(0.1)
            
        f.close()
