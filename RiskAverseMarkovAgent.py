from __future__ import print_function
from common import *
from scipy import optimize
from scipy.stats import rv_discrete
import scipy.linalg as la
import numpy as np
from copy import deepcopy

class RiskAverseMarkovAgent(Agent):
    def __init__(self, mdps, belief, max_depth=1, max_r=1, alpha=1.0, n_iter=200, K=20, n_burn_in=0, c=10):
        super(RiskAverseMarkovAgent, self).__init__()
        self.mdps = deepcopy(mdps) # identical MDPS, with different transition distributions corresponding to the support of the belief
        self.n_mdps = len(mdps)

        self.orig_belief = belief # save so that we can "reset" agent
        self.belief = np.array(belief) # this one gets updated at every timestep

        self.N_belief_updates = 1

        self.adversarial_belief = np.array(belief) # current best response to the avg performance of the MCTS policy
        self.adversarial_belief_avg = np.array(belief) # avg of past best responses
        self.adv_mixed_strategy = np.array(belief)

        self.gamma = mdps[0].gamma # discount factor extracted from MDP 0

        # The search tree is really just a dictionary, indexed by tuples (s,t) or (s,t,a)
        # items ending in h require a index ending in a state, items with ha require an index ending in an action.
        self.Wst = {} # total weight of simulations that visited state s at depth t
        self.Vst = {} # estimated value of state s and time t
        self.Wsta = {} # total weight of simulations that visity history h and took action a
        self.Wsta_br = {} # weighted number of times that action a was the best response at history h at iteration k
        self.Qsta = {} # weighted average of the total returns from simulations after history h and action a

        self.model_values = np.zeros(self.n_mdps) # avg performance of the agent under each model

        self.laplace_smoothing = 0
        self.model_counts = self.laplace_smoothing*np.ones(self.n_mdps)

        self.max_depth = max_depth # maximum depth of the search tree

        self.c = c # width of sampling at each transition node per model

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

        self.Wst = {}
        self.Vst = {}
        self.Wsat = {}
        self.Qsat = {}
        self.model_values = np.zeros(self.n_mdps)
        self.model_counts = self.laplace_smoothing*np.ones(self.n_mdps)

    # run the search to choose the best action
    def action(self, s):
        self.SparseSampling(s)
        bestq = -np.inf
        besta = -1
        for a in self.mdps[0].action_space(s):
            qval = self.Qsa[(s,a)]
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
            self.epsilon = 0.0 #0.9**(itr*50.0/self.n_iter)
            for k in range(self.K):
                rng_state = np.random.get_state()
                for mdp_i in range(self.n_mdps):
                    np.random.set_state(rng_state)
                    #w = self.adv_mixed_strategy[mdp_i]*self.n_mdps # b_adv(i)/p(i), here p(i) = 1/n_mdps
                    # playing the best response means the value estimates need not converge
                    #w = self.adversarial_belief_avg[mdp_i]*self.n_mdps
                    w = self.adversarial_belief[mdp_i]*self.n_mdps
                    V_est, V_br_eps = self.estimateV( (s,0), mdp_i, 0, self.c, w=w ) # simulate on that MDP, growing the tree in the process

                    self.model_counts[mdp_i] += 1
                    self.model_values[mdp_i] += (V_br_eps - self.model_values[mdp_i])/self.model_counts[mdp_i]

            # record current statistics for stats purposes
            self.adv_brs.append(deepcopy(self.adversarial_belief))
            self.adv_dists.append(deepcopy(self.adv_mixed_strategy))
            self.adv_avg.append(deepcopy(self.adversarial_belief_avg))
            qvals = [self.Qsta[(s,0,a)] for a in self.mdps[0].action_space(s)]
            self.agent_est_value.append(self.Vst[(s,0)])
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
    def estimateV(self, st, mdp_i, depth, c, w=1., update_tree=True):
        if depth >= self.max_depth:
            return 0,0
        if self.mdps[mdp_i].done(st[0]):
            return 0,0

        if st not in self.Wst:
            self.Wst[st] = 0
            self.Vst[st] = 0

        Qsta_est, Qsta_br_eps = self.estimateQ(st, mdp_i, depth, c, w, update_tree)
        a_star = self.greedy_action(st)
        a_eps_greedy = a_star
        if np.random.rand() < self.epsilon:
            a_eps_greedy = self.sample_rollout_action(st)


        V_est = Qsta_est[a_star]
        self.Wst[st] += 1
        self.Wsta_br[st+(a_star,)] += 1
        self.Vst[st] = self.Vst[st] + w*(V_est - self.Vst[st])/self.Wst[st]

        V_br_eps = Qsta_br_eps[a_eps_greedy]
        return V_est, V_br_eps

    def estimateQ(self, st, mdp_i, depth, c, w=1, update_tree=True):
        Qsta_est = np.zeros(len(self.mdps[mdp_i].action_space(st[0])))
        Qsta_br_eps = np.zeros(len(self.mdps[mdp_i].action_space(st[0])))

        for i,a in enumerate( self.mdps[mdp_i].action_space(st[0]) ):
            if st + (a,) not in self.Wsta:
                self.Wsta[st+(a,)] = 0
                self.Wsta_br[st+(a,)] = 0
                self.Qsta[st+(a,)] = 0

            Qsta_est[i] = 0
            Qsta_br_eps[i] = 0
            for j in range(c):
                r, sp = self.mdps[mdp_i].step(st[0], a) #TODO: replace with progressive widening for state space
                V_est, V_br_eps = self.estimateV( (sp, st[1]+1), mdp_i, depth + 1,  self.c, w=w)
                Qsta_est[i] += r + self.gamma*V_est
                Qsta_br_eps[i] += r + self.gamma*V_br_eps

            Qsta_est[i] = Qsta_est[i]*1.0/c
            Qsta_br_eps[i] = Qsta_br_eps[i]*1.0/c

            self.Wsta[st+(a,)] += 1
            self.Qsta[st+(a,)] = self.Qsta[st+(a,)] + (w*Qsta_est[i] - self.Qsta[st+(a,)])*1./self.Wsta[st+(a,)]

        return Qsta_est, Qsta_br_eps

    # the policy to use when performing rollouts
    def sample_rollout_action(self, st):
        # randomly sample action
        return np.random.choice(self.mdps[0].action_space(st[0]))

    def eps_greedy_action(self, st, eps):
        if np.random.rand() < eps:
            return self.sample_rollout_action(st)
        else:
            return self.greedy_action(st)

    def robust_action(self, st):
        best_a = -1
        best_val = 0
        if st not in self.Wst:
            return self.sample_rollout_action(st)

        for a in self.mdps[0].action_space(st[0]):
            val = self.Wsta_br[st+(a,)]

            if val > best_val:
                best_val = val
                best_a = a

        if best_a == -1:
            best_a = self.sample_rollout_action(st)

        return best_a


    def greedy_action(self, st):
        best_a = -1
        best_val = 0
        if st not in self.Wst:
            return self.sample_rollout_action(st)

        for a in self.mdps[0].action_space(st[-1]):
            val = self.Qsta[st+(a,)]

            if val > best_val:
                best_val = val
                best_a = a

        if best_a == -1:
            best_a = self.sample_rollout_action(st)

        return best_a

    def avg_action(self, h):
        s = h[-1]
        t = int(len(h)/2)
        actions = self.mdps[0].action_space(h[-1])
        if (s,t) not in self.Wst:
            import pdb; pdb.set_trace()
            probs = np.ones(len(actions))
        else:
            probs = np.array( [self.Wsta_br[(s,t,a)]*1. for a in actions] )
        probs = probs/np.sum(probs)
        a = np.random.choice(actions, p=probs)
        return a
