from __future__ import print_function
from common import *
#from sets import Set

class MultiArmedBandit(MDP):
    def __init__(self, param=0):
        self.param = param
        self.transition_probs = np.array([[[0., 0.2, 0.8],
                                          [0., 0.18, 0.82],
                                          [0., 0.14, 0.86],
                                          [0., 0., 1.]],
                                         [[0., 0.2, 0.8],
                                          [0., 0.5, 0.5],
                                          [0., 0.66, 0.34],
                                          [0., 1., 0.]]] )
        self.gamma = 1.0

    # reset environment and return s0
    def reset(self):
        return 0

    # return a list of states and transition probabilities, as NP arrays
    def transition_func(self, s, a):
        return np.arange(3), self.transition_probs[self.param][a]

    # return the reward r(s,a,sp)
    def reward_func(self, s, a, sp):
        rewards = [0,1,0]
        return rewards[sp]

    # return whether or not the current state is a terminal state
    def done(self, s):
        return s != 0

    # return a list of all the states of the MDP
    def state_space(self):
        return [0,1,2]

    # return a list of all the actions in the MDP
    def action_space(self,s):
        return [0,1,2,3]

class NPullBandit(MDP):
    def __init__(self, param=0):
        self.param = param
        # state space: [0   1   2   3   4    5    6]
        # rewards:      0  1.0 0.5 0.0 -0.1 -0.5 -1.0
        self.transition_probs = np.array( [[[0., 0., 0., 1.0, 0., 0., 0.],
                                              [0., 0., 1.0, 0., 0., 0., 0.],
                                              [0., 0.2, 0., 0., 0., 0., 0.8],
                                              [0., 0.8, 0., 0., 0., 0., 0.2]],
                                             [[0., 0., 0., 0., 1.0, 0., 0.],
                                              [0., 0., 0., 0., 0., 1.0, 0.],
                                              [0., 0.8, 0., 0., 0., 0., 0.2],
                                              [0., 0.2, 0., 0., 0., 0., 0.8]]] )
        self.gamma = 1.0

    def reset(self):
        return 0

    # return a list of states and transition probabilities, as NP arrays
    def transition_func(self, s, a):
        if s == 0:
            return np.arange(7), self.transition_probs[self.param][a]
        else:
            return np.arange(1), np.array([1.0])

    # return the reward r(s,a,sp)
    def reward_func(self, s, a, sp):
        rewards = [0, 1.0, 0.5, 0.0, -0.1, -0.5, -1.0]
        #rewards =  [0, 2.0, 1.5, 1.0, 0.9, 0.5, 0.0]
        return rewards[sp]

    # return whether or not the current state is a terminal state
    def done(self, s):
        return False

    # return a list of all the states of the MDP
    def state_space(self):
        return np.arange(7)

    # return a list of all the actions in the MDP
    def action_space(self,s):
        if s == 0:
            return np.arange(4)
        else:
            return [0]

class TreatmentPlan(MDP):
    def __init__(self, param=0):
        self.param = param
        self.allergies = np.array( [[1,0,0],
                                    [0,1,0],
                                    [0,0,1],
                                    [0,0,0]] )
        self.drug_transitions = np.array([ [[0.0, 0.4, 0.6],
                                            [0.0, 0.5, 0.5],
                                            [0.0, 0.8, 0.2]],
                                           [[0.5, 0.5, 0.0],
                                            [0.4, 0.2, 0.4],
                                            [0.2, 0.8, 0.0]] ])
        self.gamma = 1.0

    def reset(self):
        return 0

    # return a list of states and transition probabilities, as NP arrays
    def transition_func(self, s, a):
        sp_list = np.maximum( np.minimum(s + np.arange(-1, 1), 4), 0 )
        sp_dist = self.drug_transitions[self.allergies[self.param, a], a]
        return sp_list, sp_dist

    # return the reward r(s,a,sp)
    def reward_func(self, s, a, sp):
        rewards = [-1.0, -0.2, -0.2, 0.5]
        #rewards =  [0, 2.0, 1.5, 1.0, 0.9, 0.5, 0.0]
        return rewards[sp]

    # return whether or not the current state is a terminal state
    def done(self, s):
        return False

    # return a list of all the states of the MDP
    def state_space(self):
        return np.arange(7)

    # return a list of all the actions in the MDP
    def action_space(self,s):
        if s == 0:
            return np.arange(4)
        else:
            return [0]

class LavaGoalOneD(MDP):
    def __init__(self, param=0):
        self.param = param
        self.terminal_set = { -1,0, 6 }
        # self.terminal_set = { -1, 6 }
        # each 2d array corresponds to a given parameter setting
        # each columnn corresponds to moving [-2, -1, 0, 1, 2]
        # each row corresponds to probs under every action
        self.transition_probs = np.array( [ [[0.9, 0.1, 0, 0, 0],
                                           [0, 0.9, 0.1, 0, 0],
                                           [0, 0, 0.1, 0.9, 0],
                                           [0, 0, 0, 0.1, 0.9]],
                                          [[0, 0.5, 0.5, 0, 0],
                                           [0.5, 0.5, 0, 0, 0],
                                           [0, 0, 0, 0.5, 0.5],
                                           [0, 0, 0.5, 0.5, 0]],
                                          [[0, 0, 0, 0.1, 0.9],
                                           [0, 0, 0.1, 0.9, 0],
                                           [0, 0.9, 0.1, 0, 0],
                                           [0.9, 0.1, 0, 0, 0]],
                                          [[0, 0, 0.5, 0.5, 0],
                                           [0, 0, 0, 0.5, 0.5],
                                           [0.5, 0.5, 0, 0, 0],
                                           [0, 0.5, 0.5, 0, 0]] ] )
        self.gamma = 0.9


    # reset environment and return s0
    def reset(self):
        return 2

    # return a list of states and transition probabilities, as NP arrays
    def transition_func(self, s, a):
        sp = np.array([max(0,min(6,s+k)) for k in [-2,-1,0,1,2]])
        return (sp, self.transition_probs[self.param][a])

    # return the reward r(s,a,sp)
    def reward_func(self, s, a, sp):
        # reward = [-2, -0.1, -0.1, -0.1, -0.1, -0.1, +1.]
        # reward = [-2., -0.1, 0, 0.2, 0.5, 0.7, 1.0]
        reward = [-2., -0.1, 0, 0.2, 0.5, 0.7, 1.0]

        return reward[sp]

    # return whether or not the current state is a terminal state
    def done(self, s):
        return s in self.terminal_set

    # return a list of all the states of the MDP
    @property
    def state_space(self):
        return np.arange(7)

    # return a list of all the actions in the MDP
    def action_space(self,s):
        return np.arange(4)

    def render(self, s):
        string = "X-----G"
        print(string[:s] + "*" + string[s+1:])

class Inventory(MDP):
    def __init__(self, param=0):
        self.K = 4 # fixed cost of ordering
        self.c = 2 # variable cost of ordering
        self.h = lambda n: max(n, -3*n) # cost of holding n units
        self.f = lambda n: 8*n # revenue from selling n units
        self.demands = np.arange(5)
        # self.demand_models = [[0.25, 0.5, 0.25, 0., 0.],
        #                       [0.1, 0.1, 0.5, 0.3, 0.],
        #                       [0, 0.1, 0.3, 0.3, 0.3],
        #                       [0, 0.1, 0.2, 0.2, 0.5]]
        self.demand_models = np.array( [[0.25, 0.5, 0.25, 0., 0.],
                                      [0.25, 0.5, 0.25, 0., 0.],
                                      [0.25, 0.5, 0.25, 0., 0.],
                                      [0.25, 0.5, 0.25, 0., 0.]] )
        self.param = param
        self.gamma = 0.7
        self.M = 5

    # reset environment and return s0
    def reset(self):
        return 0

    # return a list of states and transition probabilities, as NP arrays
    def transition_func(self, s, a):
        sp = [min(s+a-d,self.M) for d in self.demands]
        return (sp, self.demand_models[self.param])

    # return the reward r(s,a,sp)
    def reward_func(self, s, a, sp):
        order_cost = 0
        if a > 0:
            order_cost = self.K + self.c*a;
        holding_cost = self.h(s+a)
        revenue = self.f(min(s+a - sp, s+a))
        return revenue - holding_cost - order_cost

    # return if s is terminal
    def done(self, s):
        return False

    # return a list of all the actions in the MDP
    def action_space(self,s):
        return np.arange(self.M-s)

    def render(self, s, a=0):
        print("Inventory:", s, "\tAction:", a)
