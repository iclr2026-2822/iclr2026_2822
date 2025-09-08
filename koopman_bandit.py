

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

class KoopmanBandit:
    # k_arm: # of arms in each  dimension 
    # dim dimension of the dinamic system
    # epsilon: probability for exploration in epsilon-greedy algorithm
    # initial: initial estimation for each action
    # step_size: constant step size for updating estimations
    # sample_averages: if True, use sample averages to update estimations instead of constant step size
    # UCB_param: if not None, use UCB algorithm to select action
    # gradient: if True, use gradient based bandit algorithm
    # gradient_baseline: if True, use average reward as baseline for gradient based bandit algorithm
    def __init__(self, k_arm=4,dim=2, epsilon=0., initial=0., step_size=0.1, sample_averages=False, UCB_param=None,
                 gradient=False, gradient_baseline=False, reward_func= None):
        self.k_arm= k_arm
        self.k = k_arm**dim
        self.step_size = step_size
        self.sample_averages = sample_averages
        self.indices = np.arange(self.k)
        self.time = 0
        self.UCB_param = UCB_param
        self.gradient = gradient
        self.gradient_baseline = gradient_baseline
        self.average_reward = 0
        self.true_reward = 0
        self.epsilon = epsilon
        self.initial = initial
        self.reward_func= reward_func

    def reset(self):
        # real reward for each action
        self.q_true = np.zeros(self.k) +self.true_reward

        # estimation for each action
        self.q_estimation = -10*np.ones(self.k) + self.initial

        # # of chosen times for each action
        self.action_count = np.zeros(self.k)

        

    # get an action for this bandit
    def act(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.indices)

        if self.UCB_param is not None:
            UCB_estimation = self.q_estimation + \
                     self.UCB_param * np.sqrt(np.log(self.time + 1) / (self.action_count + 1e-5))
            q_best = np.max(UCB_estimation)
            return np.random.choice([action for action, q in enumerate(UCB_estimation) if q == q_best])

        if self.gradient:
            exp_est = np.exp(self.q_estimation)
            self.action_prob = exp_est / np.sum(exp_est)
            return np.random.choice(self.indices, p=self.action_prob)

        return np.argmax(self.q_estimation)

    # take an action, update estimation for this action
    def step(self, action):
        # generate the reward under N(real reward, 1)
        if self.reward_func is not None:
            reward= self.reward_func(action, k_grid= self.k_arm)
        else:
            reward = np.random.randn() + self.q_true[action]
            
        self.time += 1
        self.average_reward = (self.time - 1.0) / self.time * self.average_reward + reward / self.time
        self.action_count[action] += 1

        if self.sample_averages:
            # update estimation using sample averages
            self.q_estimation[action] += 1.0 / self.action_count[action] * (reward - self.q_estimation[action])
        elif self.gradient:
            one_hot = np.zeros(self.k)
            one_hot[action] = 1
            if self.gradient_baseline:
                baseline = self.average_reward
            else:
                baseline = 0
            self.q_estimation = self.q_estimation + self.step_size * (reward - baseline) * (one_hot - self.action_prob)
        else:
            # update estimation with constant step size
            self.q_estimation[action] += self.step_size * (reward - self.q_estimation[action])
        return reward

def simulate(runs, time, bandits):
    
    rewards =  np.zeros((len(bandits), runs, time))
    for i, bandit in enumerate(bandits):
        for r in tqdm(range(runs)):
            bandit.reset()
            for t in range(time):
                action = bandit.act()
                reward = bandit.step(action)
                rewards[i, r, t] = reward
               
    rewards_mean = rewards.mean(axis=1)
    return rewards_mean

