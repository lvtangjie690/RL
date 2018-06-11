import sys, os
from pkg.config import config
from pkg.algos import *

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python rl_main.py int(workers)')
        sys.exit(0)
    if not os.path.exists("results/"):
        os.makedirs("results/")
    if config.ALGORITHM == 'dt':
        # only one worker for using decision tree algorithm
        workers = 1
    else:
        workers = int(sys.argv[1])
    globals()[config.ALGORITHM]().run(workers)
