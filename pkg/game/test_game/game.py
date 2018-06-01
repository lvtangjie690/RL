# coding=utf-8
import random
import argparse
from multiprocessing import Process
from .interface import Interface



reward_max = 30000
time_limit = 20
state_init = [0, 0, 0, time_limit, 0, 0, 0, 0, 0, 0, 0, 0]
state_dim  = len(state_init)
all_skills = [1, 2, 3, 4, 5, 6, 7]
action_num = len(all_skills)

# state[0] 	cp点数            state[1] 	光环
# state[2] 	临时buff          state[3] 	剩余时间
# state[4] 	技能1的冷却时间    state[5] 	技能2的冷却时间
# state[6] 	技能3的冷却时间    state[7] 	技能4的冷却时间
# state[8] 	技能5的冷却时间    state[9] 	技能6的冷却时间
# state[10] 技能7的冷却时间    state[11] 	技能1的计数点

def cast(state, action):
    state = state
    skill = all_skills[action - 1]
    state_next = state[:]

    #立即收益
    reward = 0.
    reward_fail = 0.

    #释放技能消耗的时间
    time_use = 1.0

    #连放+1cp
    if skill != 1: state_next[11] = 0.
    if skill == 1 and state_next[1 + 3] <= 0: 
        reward = 1000.
        state_next[1 + 3] = 1
        state_next[11] += 1
        if state_next[11] == 3:
            state_next[0] = min(5, state_next[0] + 1)
            state_next[11] = 0

    #长CD，高伤害+1CP
    elif skill == 2 and state_next[2 + 3] <= 0:
        reward = 2000.
        state_next[0] = min(5, state[0] + 1)
        state_next[2 + 3] = 20

    #短CD，低伤害+1CP
    elif skill == 3 and state_next[3 + 3] <=  0:
        reward = 1200
        state_next[0] = min(5, state[0] + 1)
        state_next[3 + 3] = 15

    #终结技能，CP=0
    elif skill == 4 and state_next[4 + 3] <=  0:
        reward = 1500. * pow(1.5, state[0])
        state_next[0] = 0
        state_next[4 + 3] = 15

    #纯伤害
    elif skill == 5 and state_next[5 + 3] == 0:
        reward = 2000.
        state_next[5 + 3] = 10
        
    #加光环
    elif skill == 6 and state_next[6 + 3] <=  0:
        state_next[1] =  1
        state_next[6 + 3] =  15

    #加buff，+1CP，3s
    elif skill == 7 and state_next[7 + 3] <=  0:
        reward = 1000.
        state_next[0] = min(5, state[0] + 1)
        state_next[2] = 3
        state_next[7 + 3] = 10

    else:
        # time_use = 0.5
        # reward_fail = -25
        reward_fail = -100

    # #技能miss，0.9命中率
    # if random.random() > 0.9:
    #   reward = 0
    #   state_next[0] = state[0]
    #   state_next[11] = 0

    # 计算Buff
    if state_next[1] > 0: reward *= 1.1
    if state_next[2] > 0: reward *= 2.

    # 计算时间
    for i in range(2, 11):state_next[i] = max(0., state_next[i] - time_use)

    return reward, reward_fail, state_next, state_next[3] <= 0
    #return reward * random.random() * 2, state_next
    #return reward * (random.random() + 0.5), state_next

def time(state):
    return time_limit - state[3]

def reset_agent(agent):
    agent["state"] = None
    agent["action"] = None
    agent["reward"] = None
    agent["next"] = state_init[:]
    agent["done"] = False

def send_state(agent):
    # print("send a message {}".format(agent["interface"].id))
    agent["interface"].SendSample(agent["state"], agent["action"], agent["reward"], agent["next"], agent["done"])

def cast_action(agent):
    if agent["done"]:
        print("Actor {} complete a eposide".format(agent["interface"].id)) 
        return reset_agent(agent)
    
    agent["interface"].ReceiveAction()

    agent["action"] = agent["interface"].action
    agent["state"] = agent["next"]
    reward, reward_fail, agent["next"], agent["done"] = cast(agent["state"], agent["action"])
    agent["reward"] = sum([reward_fail, reward])

def CreateAgent(id):
    agent = {"interface":Interface(id)}
    reset_agent(agent)
    while True:
        send_state(agent)
        cast_action(agent)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Launch TinyGame agent')
    parser.add_argument('-n', '--num-works', required=True, type=int, help='Number of process')
    args, unknown = parser.parse_known_args()

    for id in range(args.num_works):
        Process(target=CreateAgent, args=(id,)).start()

    # while True:
    #     for agent in agents:
    #         AgentLoop(agent)

