#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# import modules
from matplotlib import pyplot as plt
import pickle
from plot import plot_steps, plot_rewards, console_output, plot_path
from multiprocessing import Pool
from learning_algorithms import (
    discrete_SPG, discrete_MVR
)

import numpy as np

Total_Instances = 64
numthread = 16
Instances_per_Thread = int(Total_Instances/numthread)
num_episodes = 3000

run_algorithms = {
     "discrete_MVR",
}

def run_algs(ProcID):
    out_thread = []
    np.random.seed(ProcID)

    class sim_init:
        def __init__(self, num_episodes, gamma, alpha, epsilon, SGD, period,aa):
            self.num_episodes = num_episodes  # Number of training episodes
            self.gamma = gamma  # Discount rate γ 0.9
            self.alpha = alpha  # Learning rate α 0.001
            self.epsilon = epsilon  # Exploration rate ε
            self.SGD = SGD
            self.l = 750
            self.rho = 4500
            self.eps = 10 ** -3
            self.c_ = 1
            self.T_eps = 15
            self.batch_size = 1
            self.period = period
            self.aa=aa

    class sim_output:
        def __init__(self, rewards_cache, step_cache, env_cache, name_cache, std_alg_reward, std_alg_step):
            self.reward_cache = rewards_cache  # list of rewards
            self.step_cache = step_cache  # list of steps
            self.env_cache = env_cache  # list of final paths
            self.name_cache = name_cache  # list of algorithm names
            self.std_alg_reward = std_alg_reward
            self.std_alg_step = std_alg_step

        def __str__(self):
            return "# episodes: " + str(self.num_episodes) + "gamma: " + str(self.gamma) \
                + "alpha: " + str(self.alpha) + "epsilon: " + str(self.epsilon)
    for i in range(Instances_per_Thread):
        sim_output = sim_output(
            rewards_cache=[], step_cache=[], env_cache=[], name_cache=[], std_alg_reward=[], std_alg_step=[])
        SGD_reward=[]
        SGD_step=[]   
        #Run SGD
        if "discrete_MVR" in run_algorithms:
            sim_input = sim_init(num_episodes=num_episodes, gamma=0.99, alpha=0.003, epsilon=0, SGD=1, period=1000,aa=0.8)
            all_probs, sim_output, temp_goal = discrete_MVR(sim_input, sim_output)
            SGD_training_output=1
            if SGD_training_output==1:
                SGD_step = sim_output.step_cache[0]
                SGD_reward = sim_output.reward_cache[0]
        class sim_output:
            def __init__(self, rewards_cache, step_cache, env_cache, name_cache, std_alg_reward, std_alg_step):
                self.reward_cache = rewards_cache  # list of rewards
                self.step_cache = step_cache  # list of steps
                self.env_cache = env_cache  # list of final paths
                self.name_cache = name_cache  # list of algorithm names
                self.std_alg_reward=std_alg_reward
                self.std_alg_step=std_alg_step

        out_thread.append([SGD_step, SGD_reward, [SGD_training_output]])
    return out_thread

if __name__ == "__main__":
    """Learn cliff walking policies"""

    pool = Pool()  # Create a multiprocessing Pool
    data_inputs = list(range(numthread))
    with Pool(numthread) as p:
        out = p.map(run_algs, data_inputs)
    
    print('out:',out)

    SGD_step = np.zeros((1,num_episodes))
    SCRN_step = np.zeros((1,num_episodes))
    REINFORCE_step = np.zeros((1,num_episodes))

    SGD_reward = np.zeros((1,num_episodes))
    SCRN_reward = np.zeros((1,num_episodes))
    REINFORCE_reward = np.zeros((1,num_episodes))
    
    for i in range(len(out)):
        for j in range(len(out[0])):
            #SGD
            if out[i][j][0]!=[]:
               SGD_step += out[i][j][0]/np.sum(np.sum(np.array(out).reshape(numthread,Instances_per_Thread,3)[:,:,2]))
            else:
               SGD_step=SGD_step
            
            if out[i][j][1]!=[]:
               SGD_reward += out[i][j][1]/np.sum(np.sum(np.array(out).reshape(numthread,Instances_per_Thread,3)[:,:,2]))
            else:
               SGD_reward=SGD_reward
    
    class sim_output:
            def __init__(self, rewards_cache, step_cache, env_cache, name_cache, std_alg_reward, std_alg_step):
                self.reward_cache = rewards_cache  # list of rewards
                self.step_cache = step_cache  # list of steps
                self.env_cache = env_cache  # list of final paths
                self.name_cache = name_cache  # list of algorithm names
                self.std_alg_reward=std_alg_reward
                self.std_alg_step=std_alg_step
    
    #print(SGD_step)
    out0=[]
    out1=[]
    for i in range(len(out)):
        for j in range(len(out[0])):
            if out[i][j][0]!=[]:
                out0.append(out[i][j][0])
            if out[i][j][1]!=[]:
                out1.append(out[i][j][1])
    
     
    sim_SGD_output_step_std=np.std(np.array(out0),axis=0) 
    if out0==[]:
        sim_SGD_output_step_std=np.zeros(num_episodes)

    sim_SGD_output_reward_std=np.std(np.array(out1),axis=0)   
    if out1==[]:
        sim_SGD_output_reward_std=np.zeros(num_episodes)
    




    with open('MVR_step_maze.pkl', 'wb') as file:
       # A new file will be created
       pickle.dump(SGD_step[0], file)
    with open('MVR_reward_maze.pkl', 'wb') as file:
       # A new file will be created
       pickle.dump(SGD_reward[0], file)
    with open('MVR_step_maze_std.pkl', 'wb') as file:
       pickle.dump(sim_SGD_output_step_std, file)
    with open('MVR_reward_maze_std.pkl', 'wb') as file:
       # A new file will be created
       pickle.dump(sim_SGD_output_reward_std, file)

    # Plot output
    sim_out_total = sim_output(
        rewards_cache=[], step_cache=[], env_cache=[], name_cache=[],std_alg_step=[],std_alg_reward=[])
    sim_out_total.step_cache.append(SGD_step[0])
    sim_out_total.reward_cache.append(SGD_reward[0])
    sim_out_total.std_alg_step.append(sim_SGD_output_step_std)
    sim_out_total.std_alg_reward.append(sim_SGD_output_step_std)
    sim_out_total.name_cache.append("MVR")
    
    #print(SGD_step)
    #print(sim_out_total.__dict__)
    plot_steps(sim_out_total)
    plot_rewards(sim_out_total)
    print('MVR_true_instances:', np.sum(np.sum(np.array(out).reshape(numthread,Instances_per_Thread,3)[:,:,2])))
    x=np.zeros(1)
    x[0]=np.sum(np.sum(np.array(out).reshape(numthread,Instances_per_Thread,3)[:,:,2]))
    plt.plot([(i) for i in range(1)], x, color='red', label="Succussful_trajectories")
    plt.xlabel('i')
    plt.ylabel('true_instances')
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.savefig('true_instances_cliff_MVR.pdf')