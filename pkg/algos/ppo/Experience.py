class Experience(object):

    def __init__(self, state, action, prediction, reward, done, action_prob, value):
        self.state = state
        self.action = action
        self.prediction = prediction
        self.reward = reward   
        self.done = done

        self.action_prob = action_prob
        self.value = value
        self.reward_sum = 0


    @property
    def advantage(self):
        return self.reward_sum - self.value
