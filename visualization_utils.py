import numpy as np
import pandas as pd
import pickle as pkl

def create_row(entry):
    agent_params, mdp_params, reward = entry
    row = {}
    prefix = 'agent.'
    for key, val in agent_params.items():
        if type(val) is dict:
            for key2, val2 in val.items():
                row[prefix+key+'.'+key2] = val2
        elif type(val) is list:
            row[prefix+key] = [val]
        else:
            row[prefix+key] = val
    
    prefix = 'mdp.'
    for key, val in mdp_params.items():
        if type(val) is dict:
            for key2, val2 in val.items():
                row[prefix+key+'.'+key2] = val2
        elif type(val) is list:
            row[prefix+key] = [val]
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


if __name__ == "__main__":
    df = pkl_to_df("bandit_exp_test.pkl")
    
    print(df.describe())




        
        
    
    
    
    
    
    
    