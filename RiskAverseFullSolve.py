from __future__ import print_function
from common import *
from scipy import optimize
from scipy.stats import rv_discrete
import scipy.linalg as la
import numpy as np
from copy import deepcopy
from anytree import AnyNode, PostOrderIter

class RiskAverseFullSolve(Agent):
    def __init__(self, mdps, belief, max_depth=1, alpha=1.0, n_iter=200, K=20, n_burn_in=0, exp_risk=None):
        super(RiskAverseFullSolve, self).__init__()
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
        self.root = None
        self.path_to_node = {}
        
        self.model_values = np.zeros(self.n_mdps) # avg performance of the agent under each model

        self.laplace_smoothing = 0
        self.model_counts = self.laplace_smoothing*np.ones(self.n_mdps)

        self.max_depth = max_depth # maximum depth of the search tree

        self.alpha = alpha # CVaR alpha
        self.K = K # number of tree updates and model rollouts per adversarial belief update.

        self.n_iter = n_iter # number of adversarial belief updates to run the algorithm for
        self.n_burn_in = n_burn_in # start adversarial belief updates after this many iterations
        
        self.exp_risk=exp_risk

    # resets the agent to its state when constructed
    def reset(self):
        self.reset_belief()
        self.reset_tree()

    def reset_belief(self):
        self.belief = np.array(self.orig_belief)

    def reset_tree(self):
        self.N_belief_updates = 1
        self.adversarial_belief = np.array(self.orig_belief)
        
        self.root = None
        self.model_values = np.zeros(self.n_mdps)
        self.model_counts = self.laplace_smoothing*np.ones(self.n_mdps)
        
        self.terminal_histories = []

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
        self.TreeSearch(s)

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
    def TreeSearch(self, s):
        # reset tree from previous computation?
        self.adv_brs = []
        self.adv_avg = []
        self.agent_est_value = []
        self.agent_Q_vals = []
        self.adv_est_value = []
        self.model_value_history = []

        self.reset_tree()
        
        self.root = self.create_state_node(None, (s,))

        for itr in range(self.n_iter):
            self.epsilon = 0.0
            for k in range(self.K):
                rng_state = np.random.get_state() # common random numbers
                for mdp_i in range(self.n_mdps):
                    np.random.set_state(rng_state)
                    
                    w = self.adversarial_belief[mdp_i]*self.n_mdps
                    
                    V_br = self.simulate( self.root, mdp_i, w=w) # simulate all h length action choices from (s,)
                    
                    self.model_counts[mdp_i] += 1
                    self.model_values[mdp_i] += (V_br - self.model_values[mdp_i])/self.model_counts[mdp_i]

                V_est = self.estimateV( self.root ) # use counts to update value estimates
 
            # record current statistics for stats purposes
            self.adv_brs.append(deepcopy(self.adversarial_belief))
            self.adv_avg.append(deepcopy(self.adversarial_belief_avg))
            qvals = [self.path_to_node[(s,a)].V for a in self.mdps[0].action_space(s)]
            self.agent_est_value.append(self.root.V)
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
        
    def simulate(self, state_node, mdp_i, w=1.):
        state_node.N += w
        
        if int(state_node.depth / 2) >= self.max_depth:
            return 0
        if self.mdps[mdp_i].done(state_node.id[-1]):
            return 0
        
        s = state_node.id[-1]
        V_br = 0
        
        for action_node in state_node.children:
            action_node.N += w
            
            a = action_node.id[-1]                
            r, sp = self.mdps[mdp_i].step(s, a)
            
            sp_id = action_node.id + (sp,)
            if sp_id not in self.path_to_node:
                self.create_state_node(action_node, sp_id)
           
            sp_node = self.path_to_node[sp_id]
            
            Rp = self.simulate(sp_node, mdp_i, w)
            if a == self.greedy_action(state_node.id):
                action_node.N_best += w
                V_br = r + Rp
        
        return V_br
         
        
    def create_state_node(self, parent_action_node, sp_id):
        # print('Creating state node at', sp_id)
        state_node = AnyNode(N=0, V=0, id=sp_id, parent=parent_action_node)
        self.path_to_node[sp_id] = state_node
        for a in self.mdps[0].action_space(sp_id[-1]):
            ac_id = sp_id + (a,)
            ac_node = AnyNode(N=0, N_best=0, V=0, id=ac_id, parent=state_node)
            self.path_to_node[ac_id] = ac_node
        
        return state_node
    
    
    # use counts in tree along with mdps reward function to compute value function
    def estimateV(self, state_node):
        # V = max_a Q 
        if state_node.is_leaf:
            print('state node is leaf:', state_node)
            state_node.V = 0
            return 0
        
        Qs = []
        for ac_node in state_node.children:
            Qs.append(self.estimateQ(ac_node))
        
        V = max(Qs)
        
        state_node.V = V
        
        return V
    
    def estimateQ(self, action_node):
        if action_node.is_leaf:
            action_node.V = 0
            return 0
        
        counts = [] # N(s,a,sp)
        rewards = [] # R(s,a,sp)
        values = [] # V(sp)
        for sp_node in action_node.children:
            rewards.append(self.mdps[0].reward_func(sp_node.id[-3], sp_node.id[-2], sp_node.id[-1]))
            counts.append(sp_node.N)
            values.append(self.estimateV(sp_node))
            
        counts = np.array(counts)
        rewards = np.array(rewards)
        values = np.array(values)
        
        N = action_node.N
        
        if N == 0:
            Q = 0
        else:
            Q = 1./N * np.sum(counts*(rewards + values))
        
        action_node.V = Q
        
        return Q

    def random_action(self, s_id):
        # randomly sample action
        return np.random.choice(self.mdps[0].action_space(s_id[-1]))

    def greedy_action(self, s_id):
        if s_id not in self.path_to_node:
            return self.random_action(s_id)

        state_node = self.path_to_node[s_id]
        if state_node.is_leaf:
            return self.random_action(s_id)
        
        action_nodes = state_node.children
        Qvals = [ac.V for ac in action_nodes]
        best_a_idx = np.argmax(Qvals)
        
        return action_nodes[best_a_idx].id[-1]

    def avg_action(self, s_id): 
        if s_id not in self.path_to_node:
            actions = self.mdps[0].action_space(s_id[-1])
            probs = np.ones(len(actions))
        else:
            state_node = self.path_to_node[s_id]
            action_nodes = state_node.children
            actions = np.array( [ac.id[-1] for ac in action_nodes ] )
            probs = np.array( [ac.N_best for ac in action_nodes ] )
            
        probs = probs/np.sum(probs)
        a = np.random.choice(actions, p=probs)
        
        return a