import numpy as np
import pandas as pd
import pickle as pkl
from scipy import optimize
from scipy import stats
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
import matplotlib.pyplot as plt

def create_row(entry):
    agent_params, mdp_params, reward = entry
    row = {}
    prefix = 'agent.'
    for key, val in agent_params.items():
        if type(val) is dict:
            for key2, val2 in val.items():
                row[prefix+key+'.'+key2] = val2
        elif type(val) is list:
            row[prefix+key] = [tuple(val)]
        else:
            row[prefix+key] = val
    
    prefix = 'mdp.'
    for key, val in mdp_params.items():
        if type(val) is dict:
            for key2, val2 in val.items():
                row[prefix+key+'.'+key2] = val2
        elif type(val) is list:
            row[prefix+key] = [tuple(val)]
        else:
            row[prefix+key] = val
    
    row['reward'] = reward
    
    return row
    
# loads pickled data into a pandas dataframe
def pkl_to_df(filename):
    with open(filename, "rb") as f:
        data = pkl.load(f)
    df = pd.concat([pd.DataFrame(create_row(entry)) for entry in data], ignore_index=True)
    return df

#Assumes all entires in df have same belief
def plot_robustness_curve(df, filename=None):
    belief = df['agent.belief'].iloc[0]
    df = df.set_index(['agent.alpha', 'mdp.param'])
    df = df['reward']
    N_rollouts = df.groupby(['agent.alpha', 'mdp.param']).count().groupby('agent.alpha').mean()
    mean_rewards = df.groupby(['agent.alpha', 'mdp.param']).mean() # mean in each parameter setting
    var_rewards = df.groupby(['agent.alpha', 'mdp.param']).var() # var in each parameter setting
    mean_vec = mean_rewards.groupby(['agent.alpha']).apply(np.array) # combine paramter settings into vector
    cov_rewards = var_rewards.groupby(['agent.alpha']).apply(np.diag) # combine into covar matrix
    
    N_perturb_pts = 10
    perturb_amts = np.linspace(0.0,1.0,N_perturb_pts)
    agent_alphas = mean_vec.index.values
    N_alpha = len(agent_alphas)
    kl_divs = np.zeros([N_alpha,N_perturb_pts])
    perturbed_perf = np.zeros([N_alpha,N_perturb_pts])
    perturbed_perf_std = np.zeros([N_alpha,N_perturb_pts])
    
    plt.figure(num=None, figsize=(8,6), dpi=80, facecolor='w', edgecolor='k')
    for i, agent_alpha in enumerate(agent_alphas):
        mean_perf = mean_vec.loc[agent_alpha]
        cov_perf = cov_rewards.loc[agent_alpha]
        N_params = len(mean_perf)
        
        for j, alpha_perturb in enumerate(perturb_amts):
            Aeq = np.ones((1,N_params))
            beq = 1
            A1 = alpha_perturb*np.eye(N_params)
            b1 = belief
            A2 = -np.eye(N_params)
            b2 = np.zeros(N_params)
            A = np.vstack([A1,A2])
            b = np.vstack([b1, b2])
            res = optimize.linprog(mean_perf, A, b, Aeq, beq)
            kl_divs[i,j] = stats.entropy(res.x, qk=belief)
            perturbed_perf[i,j] = np.dot(res.x, mean_perf)
            perturbed_perf_std[i,j] = np.sqrt( np.dot(res.x, np.dot(cov_perf, res.x.T))/ N_rollouts.loc[agent_alpha] )

        h, = plt.plot(kl_divs[i,:], perturbed_perf[i,:], label=r"$\alpha="+str(agent_alpha)+r"$")
        plt.fill_between(kl_divs[i,:], perturbed_perf[i,:]+1.645*perturbed_perf_std[i,:], 
                         perturbed_perf[i,:]-1.645*perturbed_perf_std[i,:], color=h.get_color(), alpha=0.2)
        
    plt.legend(fontsize=16, loc=3)
    plt.grid()
    plt.xlim([0, 1.75])
    plt.xlabel(r"Error in Prior: $D_{KL}(b_{adv} || b_{prior})$", fontsize=18)
    plt.ylabel(r"Expected Reward: $\mathbb{E}_{\theta \sim b_{adv}}[J(\tau)]$", fontsize=18)
    if filename:
        plt.savefig(filename)
    plt.show()
    
    
if __name__ == "__main__":
    df = pkl_to_df("bandit_exp_test.pkl")
    
    plot_robustness_curve(df, 'test.pdf')




        
        
    
    
    
    
    
    
    