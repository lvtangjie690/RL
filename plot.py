import matplotlib.pyplot as plt
import glob
import sys
import re

colors = ['blue', 'green', 'red', 'magenta', 'cyan', \
            'darkblue', 'darkgreen', 'darkred', 'darkmagenta', 'darkcyan']

def plot_results(gamename=None, max_episode=0):
    if gamename is None:
        dir = 'results/'
    else:
        dir = 'results/' + gamename + '/'  
    plt.figure()
    for idx, filename in enumerate(glob.glob(dir+'*.txt')):
        with open(filename, 'r') as f:
            label = re.split('[\\\\/]', filename)[-1].replace('_result.txt', '')
            color = colors[idx]
            x_axis, y_axis = [], []
            for idx, line in enumerate(f.readlines()):
                episode, reward = line.split(' ')
                episode, reward = int(episode), float(reward)
                if max_episode != 0 and episode > max_episode:
                    break
                x_axis.append(episode)
                y_axis.append(reward)
            plt.plot(x_axis, y_axis, c=color, label=label)
    
    plt.legend(loc='best')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig(dir+'result.png')
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python plot.py filename max_episode')
        sys.exit(0)
    plot_results(gamename=sys.argv[1], max_episode=int(sys.argv[2]))
