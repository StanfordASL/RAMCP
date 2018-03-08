from __future__ import print_function
from common import *
from scipy import optimize
from scipy.stats import rv_discrete
import scipy.linalg as la
import numpy as np
from copy import deepcopy

class RiskAverseMCTS(Agent):
    def __init__(self, mdps, belief, max_depth=1, max_r=1, alpha=1.0, n_iter=200, K=20, n_burn_in=100):
        super(RiskAverseMCTS, self).__init__()
        self.mdps = deepcopy(mdps) # identical MDPS, with different transition distributions corresponding to the support of the belief
        self.n_mdps = len(mdps)

        self.orig_belief = belief # save so that we can "reset" agent
        self.belief = np.array(belief) # this one gets updated at every timestep

        self.N_belief_updates = 1
        self.belief_window_size = 50

        self.adversarial_belief = np.array(belief) # current best response to the avg performance of the MCTS policy
        self.adversarial_belief_avg = np.array(belief) # avg of past best responses
        self.adv_mixed_strategy = np.array(belief)

        self.gamma = mdps[0].gamma # discount factor extracted from MDP 0

        # The search tree is really just a dictionary, indexed by tuples (s0,a0,s1,a1,...)
        # items ending in h require a index ending in a state, items with ha require an index ending in an action.
        self.Wh = {} # total weight of simulations that visited history h
        self.Wha = {} # total weight of simulations that visity history h and took action a
        self.Qha = {} # weighted average of the total returns from simulations after history h and action a
        self.children_h = {} # returns indices of the visited (ha) nodes after history h
        self.children_ha = {} # returns indices of the visited (h) nodes after history h and action a

        self.model_values = np.zeros(self.n_mdps) # avg performance of the agent under each model

        self.laplace_smoothing = 0
        self.model_counts = self.laplace_smoothing*np.ones(self.n_mdps)

        self.max_depth = max_depth # maximum depth of the search tree

        # Compute the constant factor for the UCB bonus. Should be greater than the max reward possible
        self.max_r = max_r
        self.c = max_r #max_r*max_depth + 0.0000001 # For finite horizon mdps. For infinite horizon, c > max_r/(1-gamma)

        # mixing factors between best response and avg policy.
        self.eta = 1.0 # smoothing of distribution updates
        self.eta_agent = 1.0 # the changes to the UCB policy are fairly smooth, so perhaps this is not necessary / can be higher?

        self.alpha = alpha # CVaR alpha
        self.K = K # number of tree updates and model rollouts per adversarial belief update.

        self.n_iter = n_iter # number of adversarial belief updates to run the algorithm for
        self.n_burn_in = n_burn_in # start adversarial belief updates after this many iterations

    # resets the agent to its state when constructed
    def reset(self):
        self.reset_belief()
        self.reset_tree()

    def reset_belief(self):
        self.belief = np.array(self.orig_belief)

    def reset_tree(self):
        self.N_belief_updates = 1
        self.adversarial_belief = np.array(self.orig_belief)
        self.adv_mixed_strategy = np.array(self.orig_belief)

        self.Wh = {}
        self.Wha = {}
        self.Qha = {}
        self.model_values = np.zeros(self.n_mdps)
        self.model_counts = self.laplace_smoothing*np.ones(self.n_mdps)

    # run the search to choose the best action
    def action(self, s):
        self.MCTS(s)
        bestq = -np.inf
        besta = -1
        for a in self.mdps[0].action_space(s):
            qval = self.Qha[(s,a)]
            if qval > bestq:
                bestq = qval
                besta = a

        return besta
        # return self.avg_action((s,))

    def plan(self, s):
        self.MCTS(s)

    # observe a transition, update belief over mdps
    def observe(self, s, a, r, sp):
        self.update_belief(s,a,sp)

    # perform the bayesian belief update (particle filter style)
    def update_belief(self,s,a,sp):
        probs = np.zeros(self.n_mdps)
        for i, mdp in enumerate(self.mdps):
            sp_list, sp_dist = mdp.transition_func(s,a)
            #idx = np.nonzero(sp_list == sp)[0][0]
            probs[i] = sp_dist[sp_list == sp][0]#sp_dist.pmf(idx)

        belief = self.belief*probs
        self.belief = belief/np.sum(belief)
        self.adversarial_belief = deepcopy(self.belief)

    # build the tree from state s
    def MCTS(self, s):
        # reset tree from previous computation?
        self.adv_dists = []
        self.adv_brs = []
        self.adv_avg = []
        self.agent_est_value = []
        self.agent_Q_vals = []
        self.adv_est_value = []
        self.model_value_history = []

        self.reset_tree()

        for itr in range(self.n_iter):
            #self.model_counts = np.zeros(self.n_mdps) # clear before every iteration
            for k in range(self.K):
                rng_state = np.random.get_state()
                for mdp_i in range(self.n_mdps):
                    np.random.set_state(rng_state)
                    w = self.adv_mixed_strategy[mdp_i]*self.n_mdps # b_adv(i)/p(i), here p(i) = 1/n_mdps
                    # playing the best response means the value estimates need not converge
                    #w = self.adversarial_belief_avg[mdp_i]*self.n_mdps
                    #w = self.adversarial_belief[mdp_i]*self.n_mdps

                    R = self.simulate( (s,), mdp_i, 0, w=w ) # simulate on that MDP, growing the tree in the process

                    self.model_counts[mdp_i] += 1
                    self.model_values[mdp_i] += (R - self.model_values[mdp_i])/self.model_counts[mdp_i]

            # record current statistics for stats purposes
            self.adv_brs.append(deepcopy(self.adversarial_belief))
            self.adv_dists.append(deepcopy(self.adv_mixed_strategy))
            self.adv_avg.append(deepcopy(self.adversarial_belief_avg))
            qvals = [self.Qha[(s,a)] for a in self.mdps[0].action_space(s)]
            nvals = [self.Wha[(s,a)] for a in self.mdps[0].action_space(s)]
            #value = np.dot([self.Qha[(s,a)] for a in self.mdps[0].action_space(s)], [self.Wha[(s,a)] for a in self.mdps[0].action_space(s)])/self.Wh[(s,)]
            a_robust = np.argmax(nvals)
            value = qvals[a_robust]
            self.agent_est_value.append(value)
            self.agent_Q_vals.append(qvals)
            self.adv_est_value.append(np.dot(self.model_values, self.adversarial_belief_avg))
            self.model_value_history.append(deepcopy(self.model_values))

            if itr > self.n_burn_in:
                self.update_adversarial_belief()


    # solve the LP to adversarially choose a belief within the risk-polytope.
    def update_adversarial_belief(self):
        Aeq = np.ones((1,self.n_mdps))
        beq = 1
        A1 = np.eye(self.n_mdps)
        b1 = self.belief*1/self.alpha
        A2 = -np.eye(self.n_mdps)
        b2 = np.zeros(self.n_mdps)
        A = np.vstack([A1,A2])
        b = np.vstack([b1, b2])

        # do we augment this with "lower confidence bounds?"
        # should we choose adversarial belief assuming the policy will do better or worse than the mean so far?
        # assuming the worst (i.e. optimism wrt the adversary) ensures exploration
        c = np.array(self.model_values)
        # for i in range(self.n_mdps):
        #     if self.model_counts[i] != 0:
        #         c[i] -= self.c * np.sqrt(np.log(np.sum(self.model_counts))/self.model_counts[i])
        #     else:
        #         c[i] = -self.max_r

        # res = belief that minimizes the cost given a lower bound on the expected avg performance of the agent
        res = optimize.linprog(c, A, b, Aeq, beq)

        # set the adversarial belief
        self.adversarial_belief = res.x

        # compute and set the mixed strategy
        self.adv_mixed_strategy = (1-self.eta)*self.adversarial_belief_avg + self.eta*res.x

        self.N_belief_updates += 1
        # self.adversarial_belief_avg += (res.x - self.adversarial_belief_avg)/self.N_belief_updates
        self.adversarial_belief_avg += (self.adv_mixed_strategy - self.adversarial_belief_avg)/self.N_belief_updates

    # simulate a rollout under mdp_i up to depth, adding a node and updating counts if update_tree=True
    # takes a mixed strategy when choosing actions in the constructed tree
    def simulate(self, h, mdp_i, depth, w=1., update_tree=True):
        if depth >= self.max_depth:
            return 0
        if self.mdps[mdp_i].done(h[-1]):
            return 0

        if h not in self.Wh:
            for a in self.mdps[mdp_i].action_space(h[-1]): # TODO replace for continuous action space
                self.Wha[h+(a,)] = 0
                self.Qha[h+(a,)] = 0

            a = self.sample_rollout_action(h)
            r, sp = self.mdps[mdp_i].step(h[-1], a)
            R = r + self.gamma*self.rollout(h + (a,sp), mdp_i, depth + 1)
            if update_tree and w > 0:
                self.Wh[h] = w
                self.Wha[h+(a,)] = w
                self.Qha[h+(a,)] = R
            return R

        # a = self.smooth_ucb_action(h)
        depth_to_go = self.max_depth - depth
        if update_tree:
            a = self.smooth_ucb_action(h, depth_to_go=depth_to_go)
        else:
            a = self.avg_action(h)

        r, sp = self.mdps[mdp_i].step(h[-1], a) #TODO: replace with progressive widening for state space

        R = r + self.gamma*self.simulate(h + (a,sp), mdp_i, depth + 1, w=w)
        if update_tree and w>0:
            self.Wh[h] += w
            self.Wha[h+(a,)] += w
            self.Qha[h+(a,)] = self.Qha[h+(a,)] + w*(R - self.Qha[h+(a,)])*1./self.Wha[h+(a,)]
        return R

    # simulate a rollout from history h in mdp_i up to depth
    # makes no assumptions on h being in the tree
    def rollout(self, h, mdp_i, depth):
        if depth >= self.max_depth:
            return 0
        if self.mdps[mdp_i].done(h[-1]):
            return 0

        a = self.sample_rollout_action(h)
        r, sp = self.mdps[mdp_i].step(h[-1], a)
        return r + self.gamma*self.rollout(h + (a,sp), mdp_i, depth + 1)

    # the policy to use when performing rollouts
    def sample_rollout_action(self, h):
        # randomly sample action
        return np.random.choice(self.mdps[0].action_space(h[-1]))

    # choose an action at history h corresponding to the optimal UCB choice
    def ucb_action(self, h, depth_to_go=1):
        c = self.max_r*depth_to_go
        best_a = -1
        best_val = -np.inf
        for a in self.mdps[0].action_space(h[-1]):
            val = np.inf
            if self.Wha[h+(a,)] != 0 and self.Wh[h] >= 1:
                val = self.Qha[h+(a,)] + c * np.sqrt(np.log(self.Wh[h])/self.Wha[h+(a,)])

            if val > best_val:
                best_val = val
                best_a = a

        return best_a

    # choose an action at history h corresponding to the most visited
    def robust_action(self, h, depth_to_go=1):
        best_a = -1
        best_val = 0
        if h not in self.Wh:
            return self.sample_rollout_action(h)

        for a in self.mdps[0].action_space(h[-1]):
            val = self.Wha[h+(a,)]

            if val > best_val:
                best_val = val
                best_a = a

        if best_a == -1:
            best_a = self.sample_rollout_action(h)

        return best_a

    # choose an action at history h corresponding to the historical distribution of choice
    def avg_action(self, h):
        actions = self.mdps[0].action_space(h[-1])
        if h not in self.Wh:
            probs = np.ones(len(actions))
        else:
            probs = np.array( [self.Wha[h+(a,)]*1. for a in actions] ) # TODO: see if this still works with Wha rather than Nha
        probs = probs/np.sum(probs)
        a = np.random.choice(actions, p=probs)
        return a

    # perform an action at history h corresponding to a mixture of the optimal UCB action and the historical distribution
    def smooth_ucb_action(self, h, depth_to_go=1):
        z = np.random.rand()
        best_a = -1
        if z < self.eta_agent:
            # return the action given by upper confidence bounds
            return self.ucb_action(h, depth_to_go=depth_to_go)
        else:
            return self.avg_action(h)

        return best_a
