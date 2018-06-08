# a3c, ga3c, gca3c, impala, a2c
# dt, apex
ALGORITHM = 'apex'

# master's ip and port
MASTER_IP = 'localhost' 
MASTER_PORT = 4000

WORKER_BASE_PORT = 5000

# whether game push to program
GAME_PUSH_ALGORITHM = False

# use n(time_max) future reward (stop)
TIME_MAX = 5

# choose game
# Game name
# GAME_NAME = 'PointGame'
# GAME_NAME = 'GymGame'
GAME_NAME = 'TinyGame'

# if choose GymGame, select the gym env
# CartPole-v0, MountainCar-v0
GYM_GAME = 'CartPole-v0'

REWARD_MIN = -100.   # algorithm_reward will be clipped between (MIN, MAX)
REWARD_MAX = 100.
REWARD_SCALE = 5000. # algorithm_reward = game_reward / REWARD_SCALE
