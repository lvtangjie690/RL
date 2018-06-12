class Experience(object):

    def __init__(self, state, action, reward, done, next_state):
        self.state = state
        self.action = action
        self.reward = reward
        self.done = done
        self.next_state = next_state
        # absolute value of td-error
        self.priority = 0
        # important weight
        self.weight = 1.

        # index
        self.id = None


    def __eq__(self, e):
        return self.priority == e.priority


    def __lt__(self, e):
        return self.priority > e.priority
