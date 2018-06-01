class Experience(object):

    def __init__(self, state, action, prediction, reward, done):
        self.state = state
        self.action = action
        self.prediction = prediction
        self.reward = reward
        self.done = done
