import matplotlib.pyplot as plt
import glob
import sys

colors = ['blue', 'green', 'red', 'magenta', \
            'darkblue', 'darkgreen', 'darkred', 'darkmagenta']

def plot_results(max_episode=0):
    plt.figure()
    for idx, filename in enumerate(glob.glob('results/*.txt')):
        with open(filename, 'r') as f:
            label = filename.replace('_result.txt', '')
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
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        plot_results(int(sys.argv[1]))
    else:
        plot_results()
                
                  
