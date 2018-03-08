import numpy as np

class Env(object):
    def __init__(self, mdp, agent):
        self.mdp = mdp
        self.agent = agent
        self.logger = StatsLogger()

    def rollout(self, n_rollouts=1, render=True):
        self.logger.clear()
        s0 = self.mdp.reset()
        self.agent.plan(s0)
        for i in range(n_rollouts):
            # self.agent.reset()
            s0 = self.mdp.reset()
            s = s0
            h = (s0,)
            t = 0
            self.logger.init_episode()
            while (not self.mdp.done(s)) and t < self.agent.max_depth:
                a = self.agent.avg_action(h)
                #a = self.agent.action(s)
                r,sp = self.mdp.step(s,a)
                self.agent.observe(s,a,r,sp)
                self.logger.log_transition(s,a,r,sp)
                s = sp
                h = h + (a, sp)
                t = t + 1

            if render:
                self.mdp.render(s0)
                for row in self.logger.trajectories[-1]:
                    self.mdp.render(row[3], row[1])

        self.logger.process()
        return self.logger

class MDP(object):
    def __init__(self):
        self.gamma = 0.9
        pass

    def reset(self):
        raise NotImplementedError

    # return r, sp after taking action a from state s
    def step(self,s,a):
        sp_list, sp_dist = self.transition_func(s,a)
        sp = int(np.random.choice(sp_list, 1, p=sp_dist))
        r = self.reward_func(s,a,sp)
        return r, sp

    # return a tuple of (next_states, probabilities)
    def transition_func(self, s, a):
        raise NotImplementedError

    # return the reward r(s,a)
    def reward_func(self, s, a, sp):
        raise NotImplementedError

    # return whether or not the current state is a terminal state
    def done(self, s):
        raise NotImplementedError

    # return a list of all the states of the MDP
    @property
    def state_space(self):
        return []

    # return a list of all the actions in the MDP
    @property
    def action_space(self):
        raise NotImplementedError

    def render(self, s):
        pass

class Agent(object):
    def __init__(self):
        pass

    def reset(self):
        pass

    def action(self, s):
        raise NotImplementedError

    def observe(self, s, a, r, sp):
        raise NotImplementedError

class StatsLogger(object):
    def __init__(self):
        self.trajectories = []

    def init_episode(self):
        self.trajectories.append([])

    def log_transition(self, s, a, r, sp):
        self.trajectories[-1].append([s,a,r,sp])

    def process(self):
        self.total_rewards = np.zeros(len(self.trajectories))
        for i,traj in enumerate(self.trajectories):
            self.total_rewards[i] = np.sum(np.array(traj)[:,2])

    def clear(self):
        self.trajectories = []
