# -*- coding: utf-8 -*-
"""DDPGHyperParamsModel.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1v4u0UH7tjXfmqQxOmlIz9LUhhlJlDmkw
"""

# research
# https://spinningup.openai.com/en/latest/algorithms/ddpg.html

# implementation
# https://www.kaggle.com/code/auxeno/ddpg-rl

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interactive_output, FloatSlider, HBox, VBox
import ipywidgets as widgets
from collections import deque
import random

# Deterministic actor net
# state => action
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), # arbitrary middle layers
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh() # normalize to (-1, 1)
        )

    def forward(self, x):
        return self.network(x)

# Critic net evaluates potential of actions (the return of each action) taken by actor net. Used to assess action quality
# (state, action) => return
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), # takes (state, action) tuple
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # collapse to scalar return value
        )

    def forward(self, x):
        return self.network(x)

# OU noise generator class, used to add OU noise to actions
class OUNoise:
    def __init__(self, size, mu=0, sigma=0.1, theta=0.15):
        self.mu = mu * np.ones(size)
        self.sigma = sigma
        self.theta = theta
        self.size = size
        self.reset()

    # Reset noise to mean
    def reset(self):
        self.state = self.mu.copy()

    # Returns next generated value
    def sample(self):
        dx = (self.theta * (self.mu - self.state)) + (self.sigma * np.random.randn(self.size))
        self.state += dx
        return self.state.copy() # Return copy to prevent outside class edits from changing behavior

# !!! Ignore (plotting noise)
# plt.style.use('fivethirtyeight')

# def plot_ou_noise(mu, sigma, theta):
#     ou_noise = OUNoise(size=1, mu=mu, sigma=sigma, theta=theta)
#     data = [ou_noise.sample()[0] for _ in range(1000)]

#     plt.figure(figsize=(10, 5))
#     plt.plot(data, lw=2, c='#636EFA')
#     plt.title('Ornstein-Uhlenbeck Noise')
#     plt.xlabel('Time step')
#     plt.ylabel('Noise value')
#     plt.ylim(-2, 2)
#     plt.grid(True)
#     plt.show()

# # Sliders
# mu_slider = FloatSlider(value=0, min=-1, max=1, step=0.1, description='Mu:')
# sigma_slider = FloatSlider(value=0.1, min=0.01, max=0.5, step=0.01, description='Sigma:')
# theta_slider = FloatSlider(value=0.15, min=0.001, max=0.25, step=0.001, description='Theta:')

# # Interactive plot
# out = interactive_output(plot_ou_noise, {'mu': mu_slider, 'sigma': sigma_slider, 'theta': theta_slider})
# ui = HBox([mu_slider, sigma_slider, theta_slider])
# display(ui, out)

# Buffer of experience tuples D (see artcle)
class ReplayBuffer:
    def __init__(self, capacity, num_steps=1, gamma=0.99):
        self.buffer = deque(maxlen=capacity) # Deque buffer for effecient popping, pushing, and iterators from both sides
        self.num_steps = num_steps
        self.gamma = gamma
        self.n_step_buffer = deque(maxlen=num_steps)

    # Add transition tuple of type D to buffer and handle n_step logic (if needed)
    def add(self, transition):
        # transition = (s, a, r, s', terminal_state, ?)
        assert len(transition) == 6

        if self.num_steps == 1:
            state, action, reward, next_state, terminated, _ = transition
            self.buffer.append((state, action, reward, next_state, terminated)) # Append without last element of tuple to conform to D
        else:
            self.n_step_buffer.append(transition)

            # Calculate n_step reward
            _, _, _, final_state, final_termination, final_truncation = transition
            n_step_reward = 0

            for _, _, reward, _, _, _ in reversed(self.n_step_buffer):
                n_step_reward = (n_step_reward * self.gamma) + reward
            state, action, _, _, _, _ = self.n_step_buffer[0]

            # If n-step buffer is full, append to main buffer
            if len(self.n_step_buffer) == self.num_steps:
                self.buffer.append((state, action, n_step_reward, final_state, final_termination))

            # If the state is terminal, clear the n-step buffer
            if final_termination or final_truncation:
                self.n_step_buffer.clear()

    # Get sample batch of exerpiences for learner to learn from
    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return states, actions, rewards, next_states, dones


    def __len__(self):
        return len(self.buffer)

# !pip install gymnasium
import numpy as np
import torch
import random
import gymnasium as gym
import time
import math

class DDPG:
    def __init__(self, config):
        self.device = config['device']
        self.env = gym.make(config['env_name'])
        state_dim = np.prod(self.env.observation_space.shape)
        action_dim = np.prod(self.env.action_space.shape)
        self.online_actor = Actor(state_dim, action_dim, config['hidden_dim']).to(self.device)
        self.target_actor = Actor(state_dim, action_dim, config['hidden_dim']).to(self.device)
        self.online_critic = Critic(state_dim, action_dim, config['hidden_dim']).to(self.device)
        self.target_critic = Critic(state_dim, action_dim, config['hidden_dim']).to(self.device)
        self.soft_update(self.online_actor, self.target_actor, 1.)
        self.soft_update(self.online_critic, self.target_critic, 1.)
        self.optimizer_actor = torch.optim.Adam(self.online_actor.parameters(), lr=config['lr'])
        self.optimizer_critic = torch.optim.Adam(self.online_critic.parameters(), lr=config['lr'])
        self.buffer = ReplayBuffer(config['buffer_capacity'], config['num_steps'], config['gamma'])
        self.noise_generator = OUNoise(size=action_dim, mu=config['ou_mu'],
                                                      sigma=config['ou_sigma'], theta=config['ou_theta'])
        self.config = config

    # DDPG action selection
    def select_action(self, state, noise=None):
        with torch.no_grad():
            state_tensor = torch.tensor(state, device=self.device).unsqueeze(0)
            action = self.online_actor(state_tensor).squeeze(0).detach().cpu().numpy()
            if noise:
                return np.clip(action + noise, a_min=-1, a_max=1)
            return action

    # Copies the parameters from an online to target network, tau controls how fully the weights are copied.
    def soft_update(self, online, target, tau):
        for online_param, target_param in zip(online.parameters(), target.parameters()):
            target_param.data.copy_(tau * online_param.data + (1. - tau) * target_param.data)

    # Perform a single learning step
    def learn(self):
        # Sample and preprocess experience data
        states, actions, rewards, next_states, dones = self.buffer.sample(self.config['batch_size'])
        states      = torch.tensor(np.array(states),      dtype=torch.float32, device=self.device)
        actions     = torch.tensor(np.array(actions),     dtype=torch.float32, device=self.device)
        rewards     = torch.tensor(np.array(rewards),     dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones       = torch.tensor(np.array(dones),       dtype=torch.float32, device=self.device).unsqueeze(1)

        # Critic loss
        current_action_q = self.online_critic(torch.cat((states, actions), dim=1))
        with torch.no_grad():
            next_state_q = self.target_critic(torch.cat((next_states, self.online_actor(next_states)), dim=1))
            target_q = rewards + self.config['gamma'] ** self.config['num_steps'] * (1. - dones) * next_state_q
        critic_loss = F.mse_loss(current_action_q, target_q)

        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.online_critic.parameters(), self.config['grad_norm_clip'])
        self.optimizer_critic.step()

        # Actor loss
        current_action_q = self.online_critic(torch.cat((states, self.online_actor(states)), dim=1))
        actor_loss = -(current_action_q).mean()

        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.online_actor.parameters(), self.config['grad_norm_clip'])
        self.optimizer_actor.step()

    # Trains agent for a given number of steps according to given configuration
    def train(self):
        if self.config['verbose']: print("Training agent\n")

        # Logging information
        logs = {'episode_count': 0, 'episodic_reward': 0.0, 'episode_rewards': [], 'start_time': time.time()}

        # Reset episode
        state, _ = self.env.reset()

        # Main training loop
        for step in range(1, self.config['total_steps'] + 1):
            # Get action and step in envrionment
            noise = self.noise_generator.sample()
            action = self.config['action_scale'] * self.select_action(state, noise)
            next_state, reward, terminated, truncated, _ = self.env.step(action)

            # Update logs
            logs['episodic_reward'] += reward

            # Push experience to buffer
            self.buffer.add((state, action, reward, next_state, terminated, truncated))

            # Reset environment and noise
            if terminated or truncated:
                state, _ = self.env.reset()
                self.noise_generator.reset()

                # Update logs
                logs['episode_count'] += 1
                logs['episode_rewards'].append(logs['episodic_reward'])
                print(logs['episode_rewards'][-1])
                logs['episodic_reward'] = 0.0
            else:
                state = next_state

            # Perform learning step
            if len(self.buffer) > self.config['batch_size'] and step >= self.config['learning_starts']:
                self.learn()

            # Update target networks
            self.soft_update(self.online_critic, self.target_critic, self.config['tau'])
            self.soft_update(self.online_actor, self.target_actor, self.config['tau'])

            # If mean of last 20 rewards exceed target, end training
            if len(logs['episode_rewards']) > 0 and np.nanmean(logs['episode_rewards'][-20:]) >= self.config['target_reward']:
                break

            # if math.isnan(np.mean(logs['episode_rewards'][-20:])):
            #     print(logs['episode_rewards'])
            #     print(np.mean(logs['episode_rewards'][-20:]))
            #     breakpoint()

            # print(len(logs['episode_rewards']))

            # print(logs['episode_rewards'][-20:])
            # print(np.mean(logs['episode_rewards'][-20:]))
            # Print training info if verbose
            if self.config['verbose'] and step % 100 == 0 and len(logs['episode_rewards']) > 0:
                print(f"\r--- {100 * step / self.config['total_steps']:.1f}%"
                      f"\t Step: {step:,}"
                      f"\t Mean Reward: {np.nanmean(logs['episode_rewards'][-20:])}"
                      f"\t Episode: {logs['episode_count']:,}"
                      f"\t Duration: {time.time() - logs['start_time']:,.1f}s  ---", end='')
                if step % 1000 == 0:
                    print()

        # Training ended
        if self.config['verbose']: print("\n\nTraining done")
        logs['end_time'] = time.time()
        logs['duration'] = logs['end_time'] - logs['start_time']
        return logs


# ### DDPG Config ###
# ddpg_config = {
#     'env_name'       : 'Pendulum-v1',  # Environment name
#     'device'         :   'cpu',  # Device DQN runs on
#     'total_steps'    :  100000,  # Total training steps
#     'target_reward'  :    -200,  # Target reward to stop training at when reached
#     'action_scale' :        2.,  # Gym pendulum's action range is from -2 to +2 (Why OpenAI?)
#     'gamma'          :    0.99,  # Discount Factor
#     'lr'             :    1e-4,  # Learning rate
#     'hidden_dim'     :     256,  # Number of neurons in hidden layers
#     'batch_size'     :      64,  # Batch size used by learner
#     'buffer_capacity':  100000,  # Maximum replay buffer capacity
#     'tau'            :   0.001,  # Soft target network update interpolation coefficient
#     'ou_mu'          :      0.,  # OU noise mean
#     'ou_sigma'       :     0.2,  # OU noise sdev
#     'ou_theta'       :    0.15,  # OU noise reversion rate
#     'learning_starts':     512,  # Begin learning after performing this many steps
#     'num_steps'      :       3,  # Number of steps to unroll Bellman equation by
#     'grad_norm_clip' :      40,  # Global gradient clipping value
#     'verbose'        :    True,  # Verbose printing
# }


# ### Train Agent ###
# ddpg = DDPG(ddpg_config)
# logs = ddpg.train()

def plot_rewards(logs, window=5):
    rewards = logs['episode_rewards']
    moving_avg_rewards = [np.mean(rewards[max(0, i-window):i+1]) for i in range(len(rewards))]

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='Reward per Episode', c='#636EFA')
    plt.plot(moving_avg_rewards, label=f'{window}-Episode Moving Average', color='#636EFA', ls='--', alpha=0.5)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Episodic Reward')
    plt.legend()
    plt.grid(True)
    plt.show()

# print(logs)
# plot_rewards(logs, window=5)

population_size = 10
mutation_rate = 0.01
num_generations = 5

optimized_params = 'total_steps gamma lr hidden_dim batch_size buffer_capacity tau ou_mu ou_sigma ou_theta learning_starts num_steps grad_norm_clip'.split(' ')

def generate_params_from_noise_value(ou_noise_millions, ou_noise_thousands, ou_noise_hundreds, ou_noise_tens, ou_noise_digit, ou_noise_decimal):
    d = {
        'env_name'       : 'Pendulum-v1',  # Environment name
        'device'         :   'cpu',  # Device DQN runs on
        'total_steps'    :  round(ou_noise_millions.sample()[0]),  # Total training steps
        'target_reward'  :    -200,  # Target reward to stop training at when reached
        'action_scale' :        2.,  # Gym pendulum's action range is from -2 to +2 (Why OpenAI?)
        'gamma'          :    abs(ou_noise_decimal.sample()[0]),  # Discount Factor
        'lr'             :    abs(ou_noise_decimal.sample()[0]),  # Learning rate
        'hidden_dim'     :    int(round(ou_noise_hundreds.sample()[0])),  # Number of neurons in hidden layers
        'batch_size'     :    int(round(ou_noise_tens.sample()[0])),  # Batch size used by learner
        'buffer_capacity':  int(round(ou_noise_millions.sample()[0])),  # Maximum replay buffer capacity
        'tau'            :   ou_noise_decimal.sample()[0],  # Soft target network update interpolation coefficient
        'ou_mu'          :      ou_noise_decimal.sample()[0],  # OU noise mean
        'ou_sigma'       :     ou_noise_decimal.sample()[0],  # OU noise sdev
        'ou_theta'       :    ou_noise_decimal.sample()[0],  # OU noise reversion rate
        'learning_starts':     ou_noise_hundreds.sample()[0],  # Begin learning after performing this many steps
        'num_steps'      :       int(round(ou_noise_digit.sample()[0])),  # Number of steps to unroll Bellman equation by
        'grad_norm_clip' :      ou_noise_tens.sample()[0],  # Global gradient clipping value
        'verbose'        :    True,  # Verbose printing
    }
    if d['num_steps'] == 0:
        d['num_steps'] = 1
    return d

import random
def init_population(population_size):
    def generate_individual():
        def get_noise(place):
            return OUNoise(size=1, mu=int(random.random() * place))
        ou_noise_millions = get_noise(10_000_000)
        ou_noise_thousands = get_noise(10_000)
        ou_noise_hundreds = get_noise(1000)
        ou_noise_tens = get_noise(100)
        ou_noise_digit = get_noise(10)
        ou_noise_decimal = get_noise(1)

        params = generate_params_from_noise_value(ou_noise_millions, ou_noise_thousands, ou_noise_hundreds, ou_noise_tens, ou_noise_digit, ou_noise_decimal)
        return params, DDPG(params)

    result = [ generate_individual() for _ in range(population_size) ]
    print(result)
    return result

def default_params():
    return {
        'env_name'       : 'Pendulum-v1',  # Environment name
        'device'         :   'cpu',  # Device DQN runs on
        'target_reward'  :    -200,  # Target reward to stop training at when reached
        'action_scale' :        2.,  # Gym pendulum's action range is from -2 to +2 (Why OpenAI?)
        'verbose'        :    True,  # Verbose printing
    }

def crossover(parent1: DDPG, parent2: DDPG | None, mutation_rate: float) -> list[DDPG]:
    num_children = random.randint(1, 5)
    children: list[DDPG] = []
    for _ in range(num_children):
        params = default_params()
        for param in optimized_params:
            params[param] = (parent1 if random.randint(0, 1) == 1 or parent2 is None else parent2).config[param]

            if random.random() < mutation_rate:
                hyperparam_value_context = 10 * (len(str(params[param])) - 1)
                params[param] += (random.random() * 2 * hyperparam_value_context) - (hyperparam_value_context)
        children.append(DDPG(params))
    return children

import asyncio
def evolve(population_size, mutation_rate, num_generations):
    population = init_population(population_size)
    for generation in range(num_generations):
        print(f'Generation {generation + 1}')
        best_accuracy = float('-inf')
        best_individual = None, None

        async def handler(params, individual):
            logs = individual.train()
            if logs['episode_rewards'][-1] > best_accuracy:
                best_accuracy = logs['episode_rewards'][-1]
                best_individual = params, individual

        for params, individual in population:
            asyncio.create_task(handler(params, individual))

        print(f'Best reward in generation {generation + 1}: {best_accuracy}')
        print(f'Best individual params: {best_individual[0]}')

        next_generation = []
        selected = population[:population_size // 2] # Select last half of population to reproduce

        # Reproduction
        for i in range(0, len(selected), 2):
            _, parent1 = selected[i]
            _, parent2 = selected[i + 1] if i + 1 < len(selected) - 1 else (None, None)
            children = crossover(parent1, parent2, mutation_rate)
            next_generation.extend(children)

        population = next_generation

evolve(10, .01, 5)

