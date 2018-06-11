#
import numpy as np
import math
#import bisect
import json
import os

from .Config import Config

class DeltaQ(object):
    
    def __init__(self, delta_q, s, a):
        self.delta_q = delta_q
        self.s = s
        self.a = a

    def __lt__(self, r):
        return self.delta_q < r.delta_q

    def __eq__(self, r):
        return self.delta_q == r.delta_q

class Node(object):

    def get_tree(self):
        if self.is_tree():
            return self
        else:
            return self.parent.get_tree()

    def is_tree(self):
        return False

    def get_root(self):
        if self.parent.is_tree():
            return self
        else:
            return self.parent.get_root()

class DecisionNode(Node):
    """decision node decide which brach node on this state
    """

    def __init__(self, decision_idx, decision_bound, parent=None):
        self.left_node = None
        self.right_node = None

        self.decision_idx = decision_idx
        self.decision_bound = decision_bound

        self.parent = parent

    def select_node(self, state):
        if state[self.decision_idx] < self.decision_bound:
            return self.left_node
        else:
            return self.right_node
       
    def select_a_and_q(self, state):
        node = self.select_node(state)
        return node.select_a_and_q(state)

    def get_q_values(self, state):
        node = self.select_node(state)
        return node.get_q_values(state)

    def add_transition(self, transition):
        """transition = (s, a, r, done, s_n)
        """
        node = self.select_node(transition[0])
        node.add_transition(transition)

    def get_depth(self):
        return max(1+self.left_node.get_depth(), 1+self.right_node.get_depth())

    def dumps(self):
        return {'N':'D', 'L':self.left_node.dumps(), 'R':self.right_node.dumps(), 'I':str(self.decision_idx), 'B':str(self.decision_bound)}

class LeafNode(Node):

    def __init__(self, action_num, parent=None):
        self.action_num = action_num
        # self.q_values = np.array([random.random()-0.5 for _ in range(action_num)], dtype=np.float32)
        self.q_values = np.array([0. for _ in range(action_num)], dtype=np.float32)

        self.dq_his_p = [] # store delta_q >= 0 (dq, s, a)
        self.dq_his_n = [] # store delta_q < 0

        self.parent = parent
        # avg value of delta_q
        self.avg = None 

    def select_a_and_q(self, state):
        """select max_q action and max_q_value
        """
        return np.argmax(self.q_values), np.max(self.q_values)

    def get_q_values(self, state):
        return self.q_values

    def t_test(self):
        """t-test
        """
        # compute input state size
        state_size = len(self.dq_his_p[0].s) if len(self.dq_his_p) > 0 else len(self.dq_his_n[0].s)
        size_p, size_n = len(self.dq_his_p), len(self.dq_his_n)
        # only has positive or negative list
        if size_n == 0:
            miu_p, v_p = [], []
            for idx in range(state_size):
                _miu_p = sum([dq.s[idx] for dq in self.dq_his_p])/float(size_p)
                _v_p = sum([(dq.s[idx]-_miu_p)**2 for dq in self.dq_his_p])/max(1., size_p-1.)
                miu_p.append(_miu_p); v_p.append(_v_p)
            v_p = np.array(v_p)
            idx = np.argmax(v_p)
            return idx, miu_p[idx]
        elif size_p == 0:
            miu_n, v_n = [], []
            for idx in range(state_size):
                _miu_n = sum([dq.s[idx] for dq in self.dq_his_n])/float(size_n)           
                _v_n = sum([(dq.s[idx]-_miu_n)**2 for dq in self.dq_his_n])/max(1., size_n-1.)
                miu_n.append(_miu_n); v_n.append(_v_n)
            v_n = np.array(v_n)
            idx = np.argmax(v_n)
            return idx, miu_n[idx]
        # for each entry in state, compute t_test 
        miu_p = []
        miu_n = []
        t = []
        for idx in range(state_size):
            # compute avg for variable 
            _miu_p = sum([dq.s[idx] for dq in self.dq_his_p])/float(size_p)
            _miu_n = sum([dq.s[idx] for dq in self.dq_his_n])/float(size_n)
            # compute std deviation
            _v_p = sum([(dq.s[idx]-_miu_p)**2 for dq in self.dq_his_p])
            _v_n = sum([(dq.s[idx]-_miu_n)**2 for dq in self.dq_his_n])
            #print(_miu_p, _miu_n, _v_p, _v_n)
            # compute t
            if _v_n + _v_n == 0:
                _t = 0
            else:
                _t = (_miu_p-_miu_n)/math.sqrt((_v_p+_v_n)/(size_p+size_n-2)*(1./size_p + 1./size_n))
            # append miu and t to list
            miu_p.append(_miu_p); miu_n.append(_miu_n)
            t.append(abs(_t))

        t = np.array(t)
        idx = np.argmax(t)
        # if max t_test <= 0.1, stop split
        if t[idx] <= 0.1:
            return None, None
        return idx, (miu_p[idx]+miu_n[idx])/2.

    def split(self):
        """split method
        a leaf node split to two leaf nodes,
        and it changes to a decision node
        """
        decision_idx, decision_bound = self.t_test()
        if decision_idx is None:
            return
        #print('decision_idx, decision_bound', decision_idx, decision_bound)
        # create new decision node
        new_decision_node = DecisionNode(decision_idx, decision_bound, parent=self.parent)
        if not self.parent.is_tree():
            if id(self.parent.left_node) == id(self):
                self.parent.left_node = new_decision_node
            else:
                self.parent.right_node = new_decision_node
        else:
            tree = self.get_tree()
            tree.root = new_decision_node

        # create two new leaf nodes 
        # left leaf node
        left_leaf_node = LeafNode(self.action_num, parent=new_decision_node)
        left_leaf_node.q_values = self.q_values.copy()
        new_decision_node.left_node = left_leaf_node
        # right leaf node 
        right_leaf_node = LeafNode(self.action_num, parent=new_decision_node)
        right_leaf_node.q_values = self.q_values.copy()
        new_decision_node.right_node = right_leaf_node
        
    def add_transition(self, transition):
        """add delta_q, state, action into history
        """
        # compute delta_q
        s, a, r, done, s_n = transition
        next_value = 0 if done else self.get_root().select_a_and_q(s_n)[1]
        delta_q = Config.LEARNING_RATE*(r + Config.DISCOUNT*next_value - self.q_values[a])
        # if delta_q == 0:
        #     return
        # get new q_value for action a
        self.q_values[a] += delta_q
        # put s, a, delta_q into history
        dq_ = DeltaQ(delta_q, s, a)
        if delta_q >= 0:
            while len(self.dq_his_p) > Config.HISTORY_LIST_MIN_SIZE*2:
                self.dq_his_p.pop(0)
            #bisect.insort_left(self.dq_his_p, dq_)
            self.dq_his_p.append(dq_)
        else:
            while len(self.dq_his_n) > Config.HISTORY_LIST_MIN_SIZE*2:
                #self.dq_his_n.pop(-1)
                self.dq_his_n.pop(0)
            #bisect.insort_left(self.dq_his_n, dq_)
            self.dq_his_n.append(dq_)
        # check should split 
        history_size = len(self.dq_his_p) + len(self.dq_his_n)
        if history_size < Config.HISTORY_LIST_MIN_SIZE:
            return       
        # compute avg and sd
        if self.avg is None:
            self.avg = (sum([dq.delta_q for dq in self.dq_his_p]) + sum([dq.delta_q for dq in self.dq_his_n]))\
                / float(history_size)
        else:
            # compute new avg use old avg
            self.avg = self.avg + 1./history_size*(delta_q - self.avg)
        sd = math.sqrt((sum([(dq.delta_q-self.avg)**2 for dq in self.dq_his_p]) + \
                sum([(dq.delta_q-self.avg)**2 for dq in self.dq_his_n]))/(history_size-1))
        if abs(self.avg) > 2*sd:
            return
        # do split
        self.split()

    def select_node(self):
        return self

    def get_depth(self):
        return 1

    def dumps(self):
        return {'N':'L', 'Q':json.dumps(self.q_values.tolist())}

class DecisionTree(Node):
     
    def __init__(self, action_num):
        self.action_num = action_num
        self.root = LeafNode(action_num, parent=self)

    def is_tree(self):
        return True

    def add_transition(self, transition):
        """transition = (s, a, r, done, s_n)
        """
        self.root.add_transition(transition)

    def get_depth(self):
        return self.root.get_depth()

    def dumps(self):
        return self.root.dumps()

    def parse(self, dct, parent):
        if dct.get('N', 'D') == 'D':
            # decision node
            node = DecisionNode(int(dct['I']), float(dct['B']), parent=parent)
            node.left_node = self.parse(dct['L'], parent=node)
            node.right_node = self.parse(dct['R'], parent=node)
        else:
            # leaf node
            node = LeafNode(self.action_num, parent=parent)
            node.q_values = np.array(json.loads(dct['Q']), dtype=np.float32)
        return node

    def save(self):
        if not os.path.exists("dtfile/"):
            os.makedirs("dtfile/")
        with open("dtfile/dt.txt", "w") as f:
            f.write(json.dumps(self.dumps()))
            f.flush()

    def load(self):
        with open("dtfile/dt.txt", "r") as f:
            root = self.parse(json.loads(f.read()), self)
            self.root = root

    def train(self, s, a, r, done, s_n):
        self.add_transition((s, a, r, done, s_n))

    def predict(self, state):
        action, q_value = self.root.select_a_and_q(state)
        return action

    def get_q_values(self, state):
        return self.root.get_q_values(state)

class EnsembleTree(object):
     
    def __init__(self):
        self.forest = {}

    def replace_tree(self, id, tree):
        self.forest[id] = tree

    def predict(self, state):
        """vote for final action
        """
        action_counts = {}
        for tree in self.forest.values():
            action = tree.predict(state)
            action_counts[action] = action_counts.get(action, 0) + 1
        final_action, max_count = 0, 0
        for action, count in action_counts.items():
            if count > max_count:
                final_action = action
                max_count = count
        return final_action

if __name__ == "__main__":
    dt = DecisionTree() 
