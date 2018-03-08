from __future__ import print_function
from common import *
from scipy import optimize
from scipy.stats import rv_discrete
import scipy.linalg as la
import numpy as np
from copy import deepcopy

class RiskAverseSparseSampler(Agent):
    def __init__(self, mdps, belief, max_depth=1, alpha=1.0, n_iter=200, K=20, n_burn_in=0, c=10):
        super(RiskAverseSparseSampler, self).__init__()
        self.mdps = deepcopy(mdps) # identical MDPS, with different transition distributions corresponding to the support of the belief
        self.n_mdps = len(mdps)

        self.orig_belief = belief # save so that we can "reset" agent
        self.belief = np.array(belief) # this one gets updated at every timestep

        self.N_belief_updates = 1

        self.adversarial_belief = np.array(belief) # current best response to the avg performance of the MCTS policy
        self.adversarial_belief_avg = np.array(belief) # avg of past best responses

        self.gamma = mdps[0].gamma # discount factor extracted from MDP 0

        # The search tree is really just a dictionary, indexed by tuples (s0,a0,s1,a1,...)
        # items ending in h require a index ending in a state, items with ha require an index ending in an action.
        self.Wh = {} # total weight of simulations that visited history h
        self.Vh = {} # estimated value of history h
        self.Wha = {} # total weight of simulations that visity history h and took action a
        self.Wha_br = {} # weighted number of times that action a was the best response at history h at iteration k
        self.Qha = {} # weighted average of the total returns from simulations after history h and action a
        self.children_h = {} # returns indices of the visited (ha) nodes after history h
        self.children_ha = {} # returns indices of the visited (h) nodes after history h and action a

        self.model_values = np.zeros(self.n_mdps) # avg performance of the agent under each model

        self.laplace_smoothing = 0
        self.model_counts = self.laplace_smoothing*np.ones(self.n_mdps)

        self.max_depth = max_depth # maximum depth of the search tree

        self.c = c # width of sampling at each transition node per model

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
        
        self.Wh = {}
        self.Vh = {}
        self.Wha = {}
        self.Qha = {}
        self.model_values = np.zeros(self.n_mdps)
        self.model_counts = self.laplace_smoothing*np.ones(self.n_mdps)

    # run the search to choose the best action
    def action(self, s):
        self.SparseSampling(s)
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
        self.SparseSampling(s)

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
    def SparseSampling(self, s):
        # reset tree from previous computation?
        self.adv_brs = []
        self.adv_avg = []
        self.agent_est_value = []
        self.agent_Q_vals = []
        self.adv_est_value = []
        self.model_value_history = []

        self.reset_tree()

        for itr in range(self.n_iter):
            self.epsilon = 0.0
            for k in range(self.K):
                rng_state = np.random.get_state() # common random numbers
                for mdp_i in range(self.n_mdps):
                    np.random.set_state(rng_state)
                    
                    w = self.adversarial_belief[mdp_i]*self.n_mdps
                    V_est, V_br_eps = self.estimateV( (s,), mdp_i, 0, self.c, w=w ) 

                    self.model_counts[mdp_i] += 1
                    self.model_values[mdp_i] += (V_br_eps - self.model_values[mdp_i])/self.model_counts[mdp_i]

            # record current statistics for stats purposes
            self.adv_brs.append(deepcopy(self.adversarial_belief))
            self.adv_avg.append(deepcopy(self.adversarial_belief_avg))
            qvals = [self.Qha[(s,a)] for a in self.mdps[0].action_space(s)]
            self.agent_est_value.append(self.Vh[(s,)])
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

        c = np.array(self.model_values)

        # res = belief that minimizes the cost given a lower bound on the expected avg performance of the agent
        res = optimize.linprog(c, A, b, Aeq, beq)

        # set the adversarial belief
        self.adversarial_belief = res.x
        
        self.N_belief_updates += 1
        self.adversarial_belief_avg += (self.adversarial_belief - self.adversarial_belief_avg)/self.N_belief_updates

    # simulate a rollout under mdp_i up to depth, adding a node and updating counts if update_tree=True
    # takes a mixed strategy when choosing actions in the constructed tree
    def estimateV(self, h, mdp_i, depth, c, w=1., update_tree=True):
        if depth >= self.max_depth:
            return 0,0
        if self.mdps[mdp_i].done(h[-1]):
            return 0,0

        if h not in self.Wh:
            self.Wh[h] = 0
            self.Vh[h] = 0

        Qha_est, Qha_br_eps = self.estimateQ(h, mdp_i, depth, c, w, update_tree)
        a_star = self.greedy_action(h)
        a_eps_greedy = a_star
        if np.random.rand() < self.epsilon:
            a_eps_greedy = self.sample_rollout_action(h)


        V_est = Qha_est[a_star]

        self.Wh[h] += 1
        self.Wha_br[h+(a_star,)] += 1
        self.Vh[h] = self.Vh[h] + (w*V_est - self.Vh[h])*1.0/self.Wh[h]

        V_br_eps = Qha_br_eps[a_eps_greedy]
        return V_est, V_br_eps

    def estimateQ(self, h, mdp_i, depth, c, w=1, update_tree=True):
        Qha_est = np.zeros(len(self.mdps[mdp_i].action_space(h[-1])))
        Qha_br_eps = np.zeros(len(self.mdps[mdp_i].action_space(h[-1])))

        for i,a in enumerate( self.mdps[mdp_i].action_space(h[-1]) ):
            if h + (a,) not in self.Wha:
                self.Wha[h+(a,)] = 0
                self.Wha_br[h+(a,)] = 0
                self.Qha[h+(a,)] = 0

            Qha_est[i] = 0
            Qha_br_eps[i] = 0
            for j in range(c):
                r, sp = self.mdps[mdp_i].step(h[-1], a)
                V_est, V_br_eps = self.estimateV(h + (a,sp), mdp_i, depth + 1,  self.c, w=w)
                Qha_est[i] += r + self.gamma*V_est
                Qha_br_eps[i] += r + self.gamma*V_br_eps

            Qha_est[i] = Qha_est[i]*1.0/c
            Qha_br_eps[i] = Qha_br_eps[i]*1.0/c

            self.Wha[h+(a,)] += 1
            self.Qha[h+(a,)] = self.Qha[h+(a,)] + (w*Qha_est[i] - self.Qha[h+(a,)])*1./self.Wha[h+(a,)]

        return Qha_est, Qha_br_eps

    def random_action(self, h):
        # randomly sample action
        return np.random.choice(self.mdps[0].action_space(h[-1]))

    def eps_greedy_action(self, h, eps):
        if np.random.rand() < eps:
            return self.random_action(h)
        else:
            return self.greedy_action(h)

    def robust_action(self, h):
        best_a = -1
        best_val = 0
        if h not in self.Wh:
            return self.random_action(h)

        for a in self.mdps[0].action_space(h[-1]):
            val = self.Wha_star[h+(a,)]

            if val > best_val:
                best_val = val
                best_a = a

        if best_a == -1:
            best_a = self.random_action(h)

        return best_a

    def greedy_action(self, h):
        best_a = -1
        best_val = 0
        if h not in self.Wh:
            return self.random_action(h)

        for a in self.mdps[0].action_space(h[-1]):
            val = self.Qha[h+(a,)]

            if val > best_val:
                best_val = val
                best_a = a

        if best_a == -1:
            best_a = self.random_action(h)

        return best_a

    def avg_action(self, h):
        actions = self.mdps[0].action_space(h[-1])
        if h not in self.Wh:
            probs = np.ones(len(actions))
        else:
            probs = np.array( [self.Wha_br[h+(a,)]*1. for a in actions] )
        probs = probs/np.sum(probs)
        a = np.random.choice(actions, p=probs)
        return a
