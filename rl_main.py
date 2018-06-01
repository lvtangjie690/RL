import sys, os
from pkg.config import config
from pkg.algos import *

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python rl_main.py int(workers)')
        sys.exit(0)
    if not os.path.exists("results/"):
        os.makedirs("results/")
    globals()[config.ALGORITHM]().run(int(sys.argv[1]))
