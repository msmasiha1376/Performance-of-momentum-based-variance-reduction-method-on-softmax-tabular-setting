#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from mazelab import BaseEnv
from mazelab import VonNeumannMotion

import gym
from gym.spaces import Box
from gym.spaces import Discrete
from mazelab import BaseMaze
from mazelab import Object
from mazelab import DeepMindColor as color

from mazelab.generators import random_shape_maze
from mazelab.generators import random_maze
from mazelab.generators import double_t_maze

#x = random_shape_maze(width=10, height=10, max_shapes=10, max_size=2, allow_overlap=False, shape=None)
#x = random_maze(width=11, height=11, complexity=.75, density=.75)
# x = double_t_maze()
x=np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
 [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
 [1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1],
 [1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1],
 [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
 [1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1],
 [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
 [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
print(x)

start_idx = [[1, 1]]
goal_idx = [[9, 9]]
env_id = 'RandomMaze-v0'



from environment import (
    init_env,
    mark_path,
    check_game_over,
    encode_vector,
    get_state,
    get_position,
)

from actions import (
    epsilon_greedy_action,
    move_agent,
    get_max_qvalue,
    get_reward,
    compute_cum_rewards
)


import random

STATE_DIM = 11*11#10*13
ACTION_DIM = 4
size_state=11*10+11


class Maze(BaseMaze):
    @property
    def size(self):
        return x.shape
    
    def make_objects(self):
        free = Object('free', 0, color.free, False, np.stack(np.where(x == 0), axis=1))
        obstacle = Object('obstacle', 1, color.obstacle, True, np.stack(np.where(x == 1), axis=1))
        agent = Object('agent', 2, color.agent, False, [])
        goal = Object('goal', 3, color.goal, False, [])
        return free, obstacle, agent, goal
        

class Env(BaseEnv):
    def __init__(self):
        super().__init__()
        
        self.maze = Maze()
        self.motions = VonNeumannMotion()
        
        self.observation_space = Box(low=0, high=len(self.maze.objects), shape=self.maze.size, dtype=np.uint8)
        self.action_space = Discrete(len(self.motions))
        
    def step(self, action):
        motion = self.motions[action]
        current_position = self.maze.objects.agent.positions[0]
        new_position = [current_position[0] + motion[0], current_position[1] + motion[1]]
        valid = self._is_valid(new_position)
        if valid:
            self.maze.objects.agent.positions = [new_position]
        
        if self._is_goal(new_position):
            reward = +1
            done = True
        elif not valid:
            reward = -1
            done = False
        else:
            reward = -0.01
            done = False
        return self.maze.to_value(), reward, done, {}

    def get_state(self):
        size_maze = self.maze.size
        (pos_x, pos_y) = self.maze.objects.agent.positions[0]
        state = size_maze[1] * pos_x + pos_y
        return state
        
    def reset(self):
        self.maze.objects.agent.positions = start_idx
        self.maze.objects.goal.positions = goal_idx
        return self.maze.to_value()
    
    def _is_valid(self, position):
        nonnegative = position[0] >= 0 and position[1] >= 0
        within_edge = position[0] < self.maze.size[0] and position[1] < self.maze.size[1]
        passable = not self.maze.to_impassable()[position[0]][position[1]]
        return nonnegative and within_edge and passable
    
    def _is_goal(self, position):
        out = False
        for pos in self.maze.objects.goal.positions:
            if position[0] == pos[0] and position[1] == pos[1]:
                out = True
                print("===Goal is achieved!===")
                break
        return out
        
    def get_image(self):
        return self.maze.to_rgb()

gym.envs.register(id=env_id, entry_point=Env, max_episode_steps=200)
env = gym.make(env_id)




def grad_log_pi(action_trajectory, probs_trajectory):
    """
    This function computes the grad(log(pi(a|s))) for all the pair of (state, action) in the trajectory.
    Inputs:
    - action_trajectory: trajectory of actions
    - probs_trajectory: trajectory of prob. of policy taking each action
    Output:
    - grad_collection: a list of grad(log(pi(a|s))) for a given trajectory
    """

    grad_collection = []
    for t in range(len(action_trajectory)):
        action = action_trajectory[t]
        # Determine action probabilities with policy
        #  action_probs = pi(state)
        action_probs = probs_trajectory[t]

        # Encode action
        phi = encode_vector(action, ACTION_DIM)
        
        # Construct weighted state-action vector (average phi over all actions)
        weighted_phi = np.zeros((1, ACTION_DIM))

        # For demonstration only, simply copies probability vector
        for action in range(ACTION_DIM):
            action_input = encode_vector(action, ACTION_DIM)
            weighted_phi[0] += action_probs[action] * action_input[0]

        # Return grad (phi - weighted phi)
        grad = phi - weighted_phi

        grad_collection.append(grad[0])   
    return grad_collection 



def grad_trajectory(state_trajectory, action_trajectory, probs_trajectory, reward_trajectory, gamma):
    """
    This function computes the grad of objective function for a given trajectory.
    Inputs: 
    - action_trajectory: trajectory of actions
    - probs_trajectory: trajectory of prob. of policy taking each action
    - reward_trajectory: rewards of a trajectory
    - gamma: discount factor
    Output: 
    - grad: grad of obj. function for a given trajectory
    """
    grad_collection = grad_log_pi(action_trajectory, probs_trajectory)
    grad = np.zeros((STATE_DIM,ACTION_DIM))
    for t in range(len(reward_trajectory)):
        cum_reward = compute_cum_rewards(gamma, t, reward_trajectory)
        grad[state_trajectory[t],:] = grad[state_trajectory[t],:] + cum_reward*grad_collection[t]
    grad = np.reshape(grad, (1,ACTION_DIM*STATE_DIM))
    return grad, grad_collection

def grad_trajectory_prev(state_trajectory, action_trajectory, probs_trajectory, reward_trajectory, gamma, prev_state_trajectory, prev_action_trajectory,prev_probs_trajectory):
    """
    This function computes the grad of objective function for a given trajectory.
    Inputs:
    - action_trajectory: trajectory of actions
    - probs_trajectory: trajectory of prob. of policy taking each action
    - reward_trajectory: rewards of a trajectory
    - gamma: discount factor
    Output:
    - grad: grad of obj. function for a given trajectory
    """
    grad_collection = grad_log_pi(prev_action_trajectory, prev_probs_trajectory)
    grad = np.zeros((STATE_DIM,ACTION_DIM))
    for t in range(len(prev_state_trajectory)):
        cum_reward = compute_cum_rewards(gamma, t, reward_trajectory)
        grad[prev_state_trajectory[t],:] = grad[prev_state_trajectory[t],:] + cum_reward*grad_collection[t]
    grad = np.reshape(grad, (1,ACTION_DIM*STATE_DIM))
    return grad, grad_collection

def discrete_SPG(sim_input, sim_output) -> (np.array, list):
    """
    SCRN with discrete policy (manual weight updates)
    """

    num_episodes = sim_input.num_episodes
    gamma = sim_input.gamma
    Batch_size = sim_input.batch_size
    alpha0 = sim_input.alpha
    alpha = alpha0

    def softmax(theta: np.array, action_encoded: list, state: int) -> np.float:
        """Softmax function"""
        return np.exp(theta[0, state].dot(action_encoded[0]))

    def pi(state: int) -> np.array:
        """Policy: probability distribution of actions in given state"""
        probs = np.zeros(ACTION_DIM)
        for action in range(ACTION_DIM):
            action_encoded = encode_vector(action, ACTION_DIM)
            probs[action] = softmax(theta, action_encoded, state)
        return probs / np.sum(probs)


    def get_entropy_bonus(action_probs: list) -> float:
        entropy_bonus = 0
        # action_probs=action_probs.numpy()
        #  action_probs=np.squeeze(action_probs)
        for prob_action in action_probs:
            entropy_bonus -= prob_action * np.log(prob_action + 1e-5)

        return float(entropy_bonus)


    # Initialize theta
    theta = np.zeros([1, STATE_DIM, ACTION_DIM])
    count_goal_pos=np.zeros(1)
    temp_goal=np.zeros(1)

    steps_cache = np.zeros(num_episodes)
    rewards_cache = np.zeros(num_episodes)

    #Initialize grad and Hessian
    grad = np.zeros((STATE_DIM*ACTION_DIM))
    
    # Iterate over episodes
    for episode in range(num_episodes):

        if episode >= 1:
            print(episode, ":", steps_cache[episode - 1])

        # Initialize reward trajectory
        reward_trajectory = []
        action_trajectory = []
        state_trajectory = []
        probs_trajectory = []

        # Initialize environment and agent position
        env.reset()

        done = False

        while not done:

            # Get state corresponding to agent position
            state = env.get_state()


            #print((x,y))

            # Get probabilities per action from current policy
            action_probs = pi(state)

            # Select random action according to policy
            action = np.random.choice(4, p=np.squeeze(action_probs))

            # Move agent to next position
            _, reward, done, _ = env.step(action)

            (x,y) = env.maze.objects.agent.positions[0]
            
            if x == goal_idx[0][0] and y == goal_idx[0][1]:
                #print("===Goal is achieved!===")
                count_goal_pos+=1

            # Compute and store reward
            entropy_bonus = get_entropy_bonus(action_probs)
            rewards_cache[episode] += reward #+ entropy_bonus

            state_trajectory.append(state)
            action_trajectory.append(action)
            reward_trajectory.append(reward)
            probs_trajectory.append(action_probs)


            steps_cache[episode] += 1
             
        #Computing grad and Hessian for the current trajectory
        grad_traj, grad_collection_traj = grad_trajectory(state_trajectory, action_trajectory, probs_trajectory, reward_trajectory, gamma)
        grad = grad + grad_traj/Batch_size
        #print(grad)
        
        if episode%sim_input.period == 0 and episode > 0:
            alpha = alpha0/(episode/sim_input.period)

        # Update action probabilities after collecting each batch
        if episode%Batch_size == 0 and episode>0:
            #print("Batch is collected!")
            if sim_input.SGD == 1:
                Delta = alpha*grad
            Delta = np.reshape(Delta, (STATE_DIM,ACTION_DIM))
            #print(grad)
            theta = theta + Delta
            grad = np.zeros((STATE_DIM*ACTION_DIM))
            Hessian = np.zeros((STATE_DIM*ACTION_DIM, STATE_DIM*ACTION_DIM))

    if float(count_goal_pos/num_episodes) >= 0.8:
        temp_goal=1
    else:
        temp_goal=0
    print('percent_of_succuss:',count_goal_pos/num_episodes)       

    all_probs = np.zeros([STATE_DIM, ACTION_DIM])
    for state in range(48):
        action_probs = pi(state)
        all_probs[state] = action_probs

    sim_output.step_cache.append(steps_cache)
    sim_output.reward_cache.append(rewards_cache)

    sim_output.env_cache.append(env)
    if sim_input.SGD == 0:
        sim_output.name_cache.append("SCRN")
    else:
        sim_output.name_cache.append("SGD") 

    return all_probs, sim_output, temp_goal

def discrete_policy_gradient(sim_input, sim_output) -> (np.array, list):
    """
    REINFORCE with discrete policy gradient (manual weight updates)
    """

    num_episodes = sim_input.num_episodes
    gamma = sim_input.gamma
    alpha0 = sim_input.alpha
    alpha = alpha0

    def softmax(theta: np.array, action_encoded: list, state: int) -> np.float:
        """Softmax function"""
        return np.exp(theta[0, state].dot(action_encoded[0]))

    def pi(state: int) -> np.array:
        """Policy: probability distribution of actions in given state"""
        probs = np.zeros(ACTION_DIM)
        for action in range(ACTION_DIM):
            action_encoded = encode_vector(action, ACTION_DIM)
            probs[action] = softmax(theta, action_encoded, state)
        return probs / np.sum(probs)


    def get_entropy_bonus(action_probs: list) -> float:
        entropy_bonus = 0
        # action_probs=action_probs.numpy()
        #  action_probs=np.squeeze(action_probs)
        for prob_action in action_probs:
            entropy_bonus -= prob_action * np.log(prob_action + 1e-5)

        return float(entropy_bonus)

    def update_action_probabilities(
        alpha: float,
        gamma: float,
        theta: np.array,
        state_trajectory: list,
        action_trajectory: list,
        reward_trajectory: list,
        probs_trajectory: list,
    ) -> np.array:

        for t in range(len(reward_trajectory)):
            state = state_trajectory[t]
            action = action_trajectory[t]
            cum_reward = compute_cum_rewards(gamma, t, reward_trajectory)

            # Determine action probabilities with policy
            #  action_probs = pi(state)
            action_probs = probs_trajectory[t]

            # Encode action
            phi = encode_vector(action, ACTION_DIM)

            # Construct weighted state-action vector (average phi over all actions)
            weighted_phi = np.zeros((1, ACTION_DIM))

            # For demonstration only, simply copies probability vector
            for action in range(ACTION_DIM):
                action_input = encode_vector(action, ACTION_DIM)
                weighted_phi[0] += action_probs[action] * action_input[0]

            # Return score function (phi - weighted phi)
            score_function = phi - weighted_phi

            # Update theta (only update for current state, no changes for other states)
            theta[0, state] += alpha * cum_reward * score_function[0]
        return theta

    # Initialize theta
    theta = np.zeros([1, STATE_DIM, ACTION_DIM])
    count_goal_pos=np.zeros(1)
    temp_goal=np.zeros(1)
    steps_cache = np.zeros(num_episodes)
    rewards_cache = np.zeros(num_episodes)

    # Iterate over episodes
    for episode in range(num_episodes):

        if episode >= 1:
            print(episode, ":", steps_cache[episode - 1])

        # Initialize reward trajectory
        reward_trajectory = []
        action_trajectory = []
        state_trajectory = []
        probs_trajectory = []

        # Initialize environment and agent position
        env.reset()

        done = False

        while not done:

            # Get state corresponding to agent position
            state = env.get_state()


            # Get probabilities per action from current policy
            action_probs = pi(state)

            # Select random action according to policy
            action = np.random.choice(4, p=np.squeeze(action_probs))

            # Move agent to next position
            _, reward, done, _ = env.step(action)

            (x,y) = env.maze.objects.agent.positions[0]

            if x == goal_idx[0][0] and y == goal_idx[0][1]:
                #print("===Goal is achieved!===")
                count_goal_pos+=1


            # Compute and store reward
            entropy_bonus = get_entropy_bonus(action_probs)
            rewards_cache[episode] += reward #+ entropy_bonus

            state_trajectory.append(state)
            action_trajectory.append(action)
            reward_trajectory.append(reward)
            probs_trajectory.append(action_probs)


            steps_cache[episode] += 1
        
        if episode%sim_input.period == 0 and episode>0:
            alpha = alpha0/(episode/sim_input.period)

        # Update action probabilities at end of each episode
        theta = update_action_probabilities(
            alpha,
            gamma,
            theta,
            state_trajectory,
            action_trajectory,
            reward_trajectory,
            probs_trajectory,
        )
    if float(count_goal_pos/num_episodes) >= 0.8:
        temp_goal=1
    else:
        temp_goal=0
    print('percent_of_succuss:',count_goal_pos/num_episodes)  

    all_probs = np.zeros([STATE_DIM, ACTION_DIM])
    for state in range(48):
        action_probs = pi(state)
        all_probs[state] = action_probs

    sim_output.step_cache.append(steps_cache)
    sim_output.reward_cache.append(rewards_cache)

    sim_output.env_cache.append(env)
    sim_output.name_cache.append("Discrete policy gradient")

    return all_probs, sim_output, temp_goal


def discrete_MVR(sim_input, sim_output) -> (np.array, list):
    """
    MVR with discrete policy (manual weight updates)
    """

    num_episodes = sim_input.num_episodes
    gamma = sim_input.gamma
    Batch_size = sim_input.batch_size
    alpha0 = sim_input.alpha
    alpha = alpha0
    aa = sim_input.aa
    aa1=aa

    def softmax(theta: np.array, action_encoded: list, state: int) -> np.float:
        """Softmax function"""
        return np.exp(theta[0, state].dot(action_encoded[0]))

    def pi(state: int) -> np.array:
        """Policy: probability distribution of actions in given state"""
        probs = np.zeros(ACTION_DIM)
        for action in range(ACTION_DIM):
            action_encoded = encode_vector(action, ACTION_DIM)
            probs[action] = softmax(theta, action_encoded, state)
        return probs / np.sum(probs)

    def get_entropy_bonus(action_probs: list) -> float:
        entropy_bonus = 0
        # action_probs=action_probs.numpy()
        #  action_probs=np.squeeze(action_probs)
        for prob_action in action_probs:
            entropy_bonus -= prob_action * np.log(prob_action + 1e-5)

        return float(entropy_bonus)

    # Initialize theta
    theta = np.zeros([1, STATE_DIM, ACTION_DIM])
    count_goal_pos = np.zeros(1)
    temp_goal = np.zeros(1)

    steps_cache = np.zeros(num_episodes)
    rewards_cache = np.zeros(num_episodes)

    # Initialize grad and Hessian
    grad = np.zeros((STATE_DIM * ACTION_DIM))

    state_trajectory = []
    action_trajectory = []
    probs_trajectory = []
    d = np.zeros((STATE_DIM * ACTION_DIM))

    # Iterate over episodes
    for episode in range(num_episodes):

        if episode >= 1:
            print(episode, ":", steps_cache[episode - 1])


        prev_action_trajectory = action_trajectory
        prev_state_trajectory = state_trajectory
        prev_probs_trajectory = probs_trajectory

        # Initialize reward trajectory
        reward_trajectory = []
        action_trajectory = []
        state_trajectory = []
        probs_trajectory = []

        # Initialize environment and agent position
        env.reset()

        done = False

        while not done:

            # Get state corresponding to agent position
            state = env.get_state()

            # print((x,y))

            # Get probabilities per action from current policy
            action_probs = pi(state)

            # Select random action according to policy
            action = np.random.choice(4, p=np.squeeze(action_probs))

            # Move agent to next position
            _, reward, done, _ = env.step(action)

            (x, y) = env.maze.objects.agent.positions[0]

            if x == goal_idx[0][0] and y == goal_idx[0][1]:
                # print("===Goal is achieved!===")
                count_goal_pos += 1

            # Compute and store reward
            entropy_bonus = get_entropy_bonus(action_probs)
            rewards_cache[episode] += reward  # + entropy_bonus

            state_trajectory.append(state)
            action_trajectory.append(action)
            reward_trajectory.append(reward)
            probs_trajectory.append(action_probs)

            steps_cache[episode] += 1

        # Computing grad and Hessian for the current trajectory
        grad_traj, grad_collection_traj = grad_trajectory(state_trajectory, action_trajectory, probs_trajectory,
                                                          reward_trajectory, gamma)

        grad_traj_prev, grad_collection_traj_prev = grad_trajectory_prev(state_trajectory, action_trajectory,
                                                                         probs_trajectory,
                                                                         reward_trajectory, gamma,
                                                                         prev_state_trajectory, prev_action_trajectory,
                                                                         prev_probs_trajectory)
        d = (1 - aa1) * d + aa1 * grad_traj + (1 - aa1) * (grad_traj - grad_traj_prev)
        # grad = grad + grad_traj / Batch_size
        # print(grad)

        if episode % sim_input.period == 0 and episode > 0:
            alpha = alpha0 / (episode / sim_input.period)

        # Update action probabilities after collecting each batch
        if episode % Batch_size == 0 and episode > 0:
            # print("Batch is collected!")

            Delta = alpha * d
            Delta = np.reshape(Delta, (STATE_DIM, ACTION_DIM))
            # print(grad)
            theta = theta + Delta
            grad = np.zeros((STATE_DIM * ACTION_DIM))
            Hessian = np.zeros((STATE_DIM * ACTION_DIM, STATE_DIM * ACTION_DIM))

    if float(count_goal_pos / num_episodes) >= 0.8:
        temp_goal = 1
    else:
        temp_goal = 0
    print('percent_of_succuss:', count_goal_pos / num_episodes)

    all_probs = np.zeros([STATE_DIM, ACTION_DIM])
    for state in range(48):
        action_probs = pi(state)
        all_probs[state] = action_probs

    sim_output.step_cache.append(steps_cache)
    sim_output.reward_cache.append(rewards_cache)

    sim_output.env_cache.append(env)
    if sim_input.SGD == 0:
        sim_output.name_cache.append("MVR")
    else:
        sim_output.name_cache.append("SGD")

    return all_probs, sim_output, temp_goal

