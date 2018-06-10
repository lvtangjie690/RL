import random
import numpy as np
import gym

from .test_game import game as tg

class BaseGame(object):
    def __init__(self):
        pass

    def reset(self):
        """
        reset game
        @return state, 0, done, state
        """
        return None, 0, False, None
    
    def get_game_model_info(self):
        return 1, 1


    def get_state(self):
        """
        @return current state
        """
        return None

    def step(self, action):
        """
        @params action
        @return state, reward, done, next_state
        """
        return None, 0, False, None


class PointGame(BaseGame): 
    def __init__(self):
        super(PointGame, self).__init__()
        self.reset()
        self.current_state = None
        self.previous_state = None

    def get_game_model_info(self):
        """
        @return state_space_size and action_space_size
        """
        return 2, 4

    def reset(self):
        observation = np.array([(random.random()-0.5)*100, (random.random()-0.5)*100], dtype='float32')
        self.current_state = observation.copy()
        self.previous_state = observation.copy()
        self.frame_no = 0
        return self.current_state.tolist(), 0, False, self.current_state.tolist()

    def get_state(self):
        return self.current_state.tolist()

    def step(self, action):
        """@return state, reward, done, next_state
        """
        if self.current_state is None:
            self.reset()

        self.frame_no += 1
        self.previous_state = self.current_state.copy()

        def clip(value):
            return max(-10000, min(10000, value))

        def add_half(idx):
            self.current_state[idx] = clip(self.current_state[idx] + abs(self.current_state[idx]/2.0))

        def reduce_half(idx):
            self.current_state[idx] = clip(self.current_state[idx] - abs(self.current_state[idx]/2.0))

        if action == 0:
            add_half(0)
            add_half(1)
        elif action == 1:
            reduce_half(0)
            reduce_half(1) 
        elif action == 2:
            add_half(0)
            reduce_half(1)
        elif action == 3:
            reduce_half(0)
            add_half(1)

        distance = np.linalg.norm(self.current_state).item()
        reward = -distance

        done = True if distance < 0.1 or self.frame_no >= 100 else False

        return self.previous_state.tolist(), reward, done, self.current_state.tolist()


#----------------------------
#   gym game
#----------------------------

class GymGame(BaseGame):
    
    def __init__(self):
        super(GymGame, self).__init__()
        self.env = gym.make('CartPole-v0')
        self.reset()

    def reset(self):
        self.current_observation = self.env.reset().tolist()
        return self.current_observation, 0, False, self.current_observation
    
    def get_game_model_info(self):
        return self.env.observation_space.shape[0], self.env.action_space.n

    def get_state(self):
        """
        @return current state
        """
        return self.current_observation

    def step(self, action):
        """
        @params action
        @return state, reward, done, next_state
        """

        observation, reward, done, info = self.env.step(action)
        previous_observation = self.current_observation
        self.current_observation = observation.tolist()
        return previous_observation, reward, done, self.current_observation
    
    def run(self):
        print(self.get_game_model_info())
        print('-----------')
        print(self.reset())
        print('---------')
        print(self.step(0))

class TinyGame(BaseGame):
    def __init__(self):
        self.agent = {}
        self.reset()

    def reset(self):
        """
        reset game
        @return state, 0, done, state
        """
        tg.reset_agent(self.agent)
        return self.agent["state"], 0, False, self.agent["next"]
    
    def get_game_model_info(self):
        return tg.state_dim, tg.action_num

    def get_state(self):
        """
        @return current state
        """
        return self.agent["next"]

    def step(self, action):
        """
        @params action
        @return state, reward, done, next_state
        """
        self.agent["state"] = self.agent["next"]
        reward, reward_fail, self.agent["next"], done = tg.cast(self.agent["state"], action)
        return self.agent["state"], reward+reward_fail, done, self.agent["next"]

    def run(self):
        print(self.get_game_model_info())
        print('-----------')
        print(self.reset())
        print('---------')
        print(self.step(0))

if __name__ == "__main__":
    game = TinyGame()
    game.run() 
