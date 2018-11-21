from common import *
from envs import *
from copy import deepcopy
from RiskAverseMCTS import RiskAverseMCTS
from RiskAverseSparseSampler import RiskAverseSparseSampler
from RiskAverseMarkovAgent import RiskAverseMarkovAgent
from RiskAverseFullSolve import RiskAverseFullSolve

import multiprocessing
from tqdm import tqdm
import pickle

def create_mdp(params):
    return params['class'](param=params['param'])

def create_agent(params):
    mdp_list = [ params['mdp'](param=p) for p in params['mdp_params'] ]
    belief = params['belief']
    kwargs = params['kwargs']
    return params['class'](mdp_list, belief, **kwargs)

def simulate(agent_params, mdp_params, T, n_rollouts=1, verbose=False):
    # create agent
    agent = create_agent(agent_params)
    mdp = create_mdp(mdp_params)
    replan = agent_params['replan']

    rewards = []
    
    for i in range(n_rollouts):
        s = mdp.reset()
        h = (s,)
        if replan or i == 0:
            agent.plan(s)
            if verbose:
                print('Planning...', s)

        total_r = 0

        for t in range(T):
            a = agent.avg_action(h)
            if verbose:
                print('Belief is', agent.belief)
                print('Taking action a =', a, 'from s = ', s)

            r,sp = mdp.step(s,a)
            agent.observe(s,a,r,sp)
            total_r += r

            s = sp
            h = h + (a,sp)

            if mdp.done(s):
                break

            if replan and t < T - 1:
                agent.max_depth -= 1
                agent.plan(s)
                h = (s,)
                if verbose:
                    print('Planning...')
        
        rewards.append(total_r)
    
    return [ (agent_params, mdp_params, r) for r in rewards ]

def simulate_(params):
    return simulate(*params)

def run_batch_sims(params_gen, filename, N_workers=1):
    results = []
    with multiprocessing.Pool(N_workers) as p:
        generator = params_gen()
        for res in tqdm(p.imap(simulate_, generator), total=len(generator), position=0):
            results += res

    with open(filename, "wb") as f:
        pickle.dump(results, f)

class BanditExperimentGen(object):
    def __init__(self):
        self.agent_base = {
            'class': RiskAverseSparseSampler,
            'mdp': NPullBandit,
            'mdp_params': [0,1],
            'belief': [0.6, 0.4],
            'kwargs': {
                'max_depth': 4,
                'alpha': 1.0,
                'n_iter': 1250,
                'K': 5,
                'c': 1
            },
            'replan': False,
        }

        self.mdp_base = {
            'class': NPullBandit,
            'param': 0
        }

        self.T = 4
        self.N_params = 2
        self.N_alpha = 3
        self.N_rollouts = 1000
        
    def __len__(self):
        return self.N_alpha*self.N_params
    
    def __iter__(self):
        return self.gen()
    
    def gen(self):
        for param in range(self.N_params):
            for alpha in np.linspace(0.25,1.0,self.N_alpha):
                agent_param = deepcopy(self.agent_base)
                mdp_param = deepcopy(self.mdp_base)

                agent_param['kwargs']['alpha'] = alpha
                mdp_param['param'] = param
                yield (agent_param, mdp_param, self.T, self.N_rollouts)
                
                
class FullSolveBanditExperimentGen(object):
    def __init__(self):
        self.agent_base = {
            'class': RiskAverseFullSolve,
            'mdp': NPullBandit,
            'mdp_params': [0,1],
            'belief': [0.6, 0.4],
            'kwargs': {
                'max_depth': 4,
                'alpha': 1.0,
                'n_iter': 1250,
                'K': 1,
            },
            'replan': False,
        }

        self.mdp_base = {
            'class': NPullBandit,
            'param': 0
        }

        self.T = 4
        self.N_params = 2
        self.N_alpha = 3
        self.N_rollouts = 1000
        
    def __len__(self):
        return self.N_alpha*self.N_params
    
    def __iter__(self):
        return self.gen()
    
    def gen(self):
        for param in range(self.N_params):
            for alpha in np.linspace(0.25,1.0,self.N_alpha):
                agent_param = deepcopy(self.agent_base)
                mdp_param = deepcopy(self.mdp_base)

                agent_param['kwargs']['alpha'] = alpha
                mdp_param['param'] = param
                yield (agent_param, mdp_param, self.T, self.N_rollouts)


class FullSolveTreatmentExperimentGen(object):
    def __init__(self):
        self.agent_base = {
            'class': RiskAverseFullSolve,
            'mdp': TreatmentPlan,
            'mdp_params': list(range(15)),
            'belief': [0.25] + [0.75/14]*14,
            'kwargs': {
                'max_depth': 4,
                'alpha': 1.0,
                'n_iter': 2500,
                'K': 1
            },
            'replan': False,
        }

        self.mdp_base = {
            'class': TreatmentPlan,
            'param': 0
        }

        self.T = 4
        self.N_params = 15
        self.N_alpha = 3
        self.N_rollouts = 500
        
    def __len__(self):
        return self.N_alpha*self.N_params*2
    
    def __iter__(self):
        return self.gen()
    
    def gen(self):
        for param in range(self.N_params):
            for alpha in np.linspace(0.2,1.0,self.N_alpha):
                agent_param = deepcopy(self.agent_base)
                mdp_param = deepcopy(self.mdp_base)

                agent_param['kwargs']['alpha'] = alpha
                mdp_param['param'] = param
                yield (agent_param, mdp_param, self.T, self.N_rollouts)
                
                    
class TreatmentExperimentGen(object):
    def __init__(self):
        self.agent_base = {
            'class': RiskAverseSparseSampler,
            'mdp': TreatmentPlan,
            'mdp_params': list(range(15)),
            'belief': [0.25] + [0.75/14]*14,
            'kwargs': {
                'max_depth': 4,
                'alpha': 1.0,
                'n_iter': 2500,
                'K': 5,
                'c': 1
            },
            'replan': False,
        }

        self.mdp_base = {
            'class': TreatmentPlan,
            'param': 0
        }

        self.T = 4
        self.N_params = 15
        self.N_alpha = 3
        self.N_rollouts = 500
        
    def __len__(self):
        return self.N_alpha*self.N_params*2
    
    def __iter__(self):
        return self.gen()
    
    def gen(self):
        for param in range(self.N_params):
            for alpha in np.linspace(0.2,1.0,self.N_alpha):
                agent_param = deepcopy(self.agent_base)
                mdp_param = deepcopy(self.mdp_base)

                agent_param['kwargs']['alpha'] = alpha
                mdp_param['param'] = param
                yield (agent_param, mdp_param, self.T, self.N_rollouts)
                
            for exp_risk in np.linspace(0.1,1.5,self.N_alpha):
                agent_param = deepcopy(self.agent_base)
                mdp_param = deepcopy(self.mdp_base)

                agent_param['kwargs']['exp_risk'] = exp_risk
                mdp_param['param'] = param
                yield (agent_param, mdp_param, self.T, self.N_rollouts)
                    
if __name__ == "__main__":
#     agent_params = {
#         'class': RiskAverseSparseSampler,
#         'mdp': NPullBandit,
#         'mdp_params': [0,1],
#         'belief': [0.6, 0.4],
#         'kwargs': {
#             'max_depth': 4,
#             'alpha': 1.0,
#             'n_iter': 300,
#             'K': 5,
#             'c': 1
#         },
#         'replan': False,
#     }
    
#     mdp_params = {
#         'class': NPullBandit,
#         'param': 0
#     }
#     import cProfile    
#     result = cProfile.run("simulate(agent_params, mdp_params, 4, verbose=True)")
    
#     print(result)
    run_batch_sims(TreatmentExperimentGen, "treatment_ramcp_run6_initat3.pkl", 10)