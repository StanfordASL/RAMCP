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

def plot_spreads(df, df_rmcp=None, filename=None):
    belief = df['agent.belief'].iloc[0]
    df = df.set_index(['agent.kwargs.alpha', 'mdp.param'])
    df = df['reward']
    mean_rewards = df.groupby(['agent.kwargs.alpha', 'mdp.param']).mean() # mean in each parameter setting
    var_rewards = df.groupby(['agent.kwargs.alpha', 'mdp.param']).var() # var in each parameter setting
    mean_vec = mean_rewards.groupby(['agent.kwargs.alpha']).apply(np.array) # combine paramter settings into vector
    cov_rewards = var_rewards.groupby(['agent.kwargs.alpha']).apply(np.diag) # combine into covar 
    
    alphas = mean_vec.index.values
    means = mean_vec.apply(lambda x: np.dot(x, belief))
    stds = mean_vec.apply(lambda x: np.cov(x, aweights=belief)).apply(np.sqrt)
    
    if df_rmcp is not None:
        df = df_rmcp.set_index(['agent.kwargs.alpha', 'mdp.param'])
        df = df['reward']
        mean_rewards = df.groupby(['agent.kwargs.alpha', 'mdp.param']).mean() # mean in each parameter setting
        var_rewards = df.groupby(['agent.kwargs.alpha', 'mdp.param']).var() # var in each parameter setting
        mean_vec = mean_rewards.groupby(['agent.kwargs.alpha']).apply(np.array) # combine paramter settings into vector
        cov_rewards = var_rewards.groupby(['agent.kwargs.alpha']).apply(np.diag) # combine into covar 

        alphas_rmcp = mean_vec.index.values
        means_rmcp = mean_vec.apply(lambda x: np.dot(x, belief))
        stds_rmcp = mean_vec.apply(lambda x: np.cov(x, aweights=belief)).apply(np.sqrt)
    
    plt.figure(num=None, figsize=(8,6), dpi=80, facecolor='w', edgecolor='k')
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.errorbar(alphas[::2], means.values[::2], yerr=1.96*stds.values[::2], marker='o', elinewidth=3, markersize=8, capthick=2, capsize=5, linestyle='None', label=r"\textbf{RAMCP}")
    if df_rmcp is not None:
        plt.errorbar(alphas_rmcp, means_rmcp.values, yerr=1.96*stds_rmcp, marker='o', elinewidth=3, markersize=8, capthick=2, capsize=5, linestyle='None', mfc='white', label=r"RMCP")
        plt.legend(fontsize=16, loc=3)
    plt.grid()
    plt.xlim([0.15, 1.05])
    plt.ylim([0, 2.5])
    plt.xlabel(r"CVaR Quantile ($\alpha$)", fontsize=18)
    plt.ylabel(r"Reward", fontsize=18)
    
    plt.text(0.15, -0.28, 'Worst Case', horizontalalignment='left', fontsize=15)
    plt.text(1.05, -0.28, 'Risk Neutral', horizontalalignment='right', fontsize=15)
    
    if filename:
        plt.savefig(filename)
    plt.show()
        
    
#Assumes all entires in df have same belief
def plot_robustness_curve(df, filename=None):
    belief = df['agent.belief'].iloc[0]
    df = df.set_index(['agent.kwargs.alpha', 'mdp.param'])
    df = df['reward']
    N_rollouts = df.groupby(['agent.kwargs.alpha', 'mdp.param']).count().groupby('agent.kwargs.alpha').mean()
    mean_rewards = df.groupby(['agent.kwargs.alpha', 'mdp.param']).mean() # mean in each parameter setting
    var_rewards = df.groupby(['agent.kwargs.alpha', 'mdp.param']).var() # var in each parameter setting
    mean_vec = mean_rewards.groupby(['agent.kwargs.alpha']).apply(np.array) # combine paramter settings into vector
    cov_rewards = var_rewards.groupby(['agent.kwargs.alpha']).apply(np.diag) # combine into covar matrix
    
    N_perturb_pts = 10
    perturb_amts = np.linspace(0.0,1.0,N_perturb_pts)
    agent_alphas = mean_vec.index.values
    N_alpha = len(agent_alphas)
    kl_divs = np.zeros([N_alpha,N_perturb_pts+1])
    perturbed_perf = np.zeros([N_alpha,N_perturb_pts+1])
    perturbed_perf_std = np.zeros([N_alpha,N_perturb_pts+1])
    
    plt.figure(num=None, figsize=(8,6), dpi=80, facecolor='w', edgecolor='k')
    plt.tick_params(axis='both', which='major', labelsize=14)
    for i, agent_alpha in enumerate(agent_alphas):
        mean_perf = mean_vec.loc[agent_alpha]
        cov_perf = cov_rewards.loc[agent_alpha]
        N_params = len(mean_perf)
        
        for j, alpha_perturb in enumerate(reversed(perturb_amts)):
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
        
        kl_divs[i,-1] = max(kl_divs[i,-1], 1.75)
        perturbed_perf[i,-1] = perturbed_perf[i,-2]
        perturbed_perf_std[i,-1] = perturbed_perf_std[i,-2]
        if i %2 == 0: # or i == 3 or i == 6:
            h, = plt.plot(kl_divs[i,:], perturbed_perf[i,:], label=r"$\alpha="+str(agent_alpha)+r"$")
            plt.fill_between(kl_divs[i,:], perturbed_perf[i,:]+1.645*perturbed_perf_std[i,:], 
                             perturbed_perf[i,:]-1.645*perturbed_perf_std[i,:], color=h.get_color(), alpha=0.15, linestyle=':', edgecolor='black', linewidth=2)
        
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




        
        
    
    
    
    
    
    
    