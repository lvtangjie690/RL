class Experience(object):

    def __init__(self, state, action, prediction, reward, done, value):
        self.state = state
        self.action = action
        self.prediction = prediction
        self.reward = reward
        self.done = done
        self.value = value

        self.second_sum = 0
        self.reward_sum = 0
        self.v = 0
        self.rho = 0


    @property
    def miu(self):
        return self.prediction[self.action]
