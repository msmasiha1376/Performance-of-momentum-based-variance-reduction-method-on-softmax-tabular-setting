from matplotlib import pyplot as plt
from plot import plot_steps, plot_rewards, console_output, plot_path
import pickle


import numpy as np
num_episodes = 10000
SGD_step = np.zeros((1,num_episodes))
MVR_step = np.zeros((1,num_episodes))
REINFORCE_step = np.zeros((1,num_episodes))
REINFORCE_step_ent = np.zeros((1,num_episodes))
SGD_step_ent = np.zeros((1,num_episodes))
SGD_reward = np.zeros((1,num_episodes))
MVR_reward = np.zeros((1,num_episodes))
REINFORCE_reward = np.zeros((1,num_episodes))  
SGD_reward_ent = np.zeros((1,num_episodes))
REINFORCE_reward_ent = np.zeros((1,num_episodes))      

class sim_output:
           def __init__(self, rewards_cache, step_cache, env_cache, name_cache, std_alg_reward, std_alg_step):
              self.reward_cache = rewards_cache  # list of rewards
              self.step_cache = step_cache  # list of steps
              self.env_cache = env_cache  # list of final paths
              self.name_cache = name_cache  # list of algorithm names
              self.std_alg_reward=std_alg_reward
              self.std_alg_step=std_alg_step



with open('REINFORCE_step.pkl', 'rb') as file:    
        # Call load method to deserialze
        REINFORCE_step[0] = pickle.load(file)
with open('REINFORCE_reward.pkl', 'rb') as file:
      
        # Call load method to deserialze
        REINFORCE_reward[0] = pickle.load(file)
with open('SGD_step.pkl', 'rb') as file:
        # Call load method to deserialze
        SGD_step[0] = pickle.load(file)
with open('SGD_reward.pkl', 'rb') as file:
        # Call load method to deserialze
        SGD_reward[0] = pickle.load(file)
with open('MVR_step.pkl', 'rb') as file:
        # Call load method to deserialze
        MVR_step[0] = pickle.load(file)
with open('MVR_reward.pkl', 'rb') as file:
        # Call load method to deserialze
        MVR_reward[0] = pickle.load(file)
with open('MVR_step_std.pkl', 'rb') as file: 
        # Call load method to deserialze
    sim_MVR_output_step_std = pickle.load(file)
with open('MVR_reward_std.pkl', 'rb') as file: 
        # Call load method to deserialze
    sim_MVR_output_reward_std = pickle.load(file)
with open('SGD_step_std.pkl', 'rb') as file:     
        # Call load method to deserialze
    sim_SGD_output_step_std = pickle.load(file)
with open('SGD_reward_std.pkl', 'rb') as file:
        # Call load method to deserialze
    sim_SGD_output_reward_std = pickle.load(file)
with open('REINFORCE_step_std.pkl', 'rb') as file:     
        # Call load method to deserialze
    sim_DPG_output_step_std = pickle.load(file)
with open('REINFORCE_reward_std.pkl', 'rb') as file:
        # Call load method to deserialze
    sim_DPG_output_reward_std = pickle.load(file)
#################################    

sim_out_total = sim_output(
        rewards_cache=[], step_cache=[], env_cache=[], name_cache=[],std_alg_step=[],std_alg_reward=[])
sim_out_total.step_cache.append(SGD_step[0])
sim_out_total.reward_cache.append(SGD_reward[0])
sim_out_total.std_alg_step.append(sim_SGD_output_step_std)
sim_out_total.std_alg_reward.append(sim_SGD_output_step_std)
sim_out_total.name_cache.append("SPG")

sim_out_total.step_cache.append(REINFORCE_step[0])
sim_out_total.reward_cache.append(REINFORCE_reward[0])
sim_out_total.std_alg_step.append(sim_DPG_output_step_std)
sim_out_total.std_alg_reward.append(sim_DPG_output_reward_std)
sim_out_total.name_cache.append("REINFORCE")  

sim_out_total.step_cache.append(MVR_step[0])
sim_out_total.reward_cache.append(MVR_reward[0])
sim_out_total.std_alg_step.append(sim_MVR_output_step_std)
sim_out_total.std_alg_reward.append(sim_MVR_output_reward_std)
sim_out_total.name_cache.append("MVR")
#print(SGD_step)
#print(sim_out_total.__dict__)
plot_steps(sim_out_total)
plot_rewards(sim_out_total)    