import gym
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch import nn
from torch import optim
from torch.distributions.categorical import Categorical
import random
from mlagents_envs.environment import UnityEnvironment


sns.set()

class ActorCriticNN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU())

        self.policy_layers = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64,action_dim))

        self.value_layers = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1))

    def value(self, obs):
        z = self.shared_layers(obs)
        value = self.value_layers(z)
        return value

    def policy(self, obs):
        z = self.shared_layers(obs)
        policy_logits = self.policy_layers(z)
        return policy_logits

    def forward(self, obs):
        z = self.shared_layers(obs)
        policy_logits = self.policy_layers(z)
        value = self.value_layers(z)
        return policy_logits, value

#trainer
class PPOTrainer():
    def __init__(self, actor_critic, ppo_clip_val, target_kl_div, policy_lr, value_lr, max_policy_train_iters = 80, value_train_iters = 80):
        self.ac = actor_critic
        self.ppo_clip_val = ppo_clip_val
        self.target_kl_div = target_kl_div
        self.max_policy_train_iters = max_policy_train_iters
        self.value_train_iters = value_train_iters
    
        policy_params = list(self.ac.shared_layers.parameters()) + \
            list(self.ac.policy_layers.parameters())
        self.policy_optim = optim.Adam(policy_params, lr = policy_lr)
    
        value_params = list(self.ac.shared_layers.parameters()) + \
            list(self.ac.value_layers.parameters())
        self.value_optim = optim.Adam(value_params, lr = value_lr)

    def train_policy(self, obs, acts, old_log_probs, gaes):

        for _ in range(self.max_policy_train_iters):
                
            self.policy_optim.zero_grad()
    
            new_logits = self.ac.policy(obs)
            new_logits = Categorical(logits = new_logits)
            new_log_probs = new_logits.log_prob(acts)
    
            policy_ratio = torch.exp(new_log_probs - old_log_probs)
            clipped_ratio = policy_ratio.clamp(1 - self.ppo_clip_val, 1 + self.ppo_clip_val)
            clipped_loss = clipped_ratio * gaes
            full_loss = policy_ratio * gaes
            policy_loss = -torch.min(full_loss, clipped_loss).mean()
    
            policy_loss.backward()
            self.policy_optim.step()

            kl_div = (old_log_probs - new_log_probs).mean()
            if kl_div >= self.target_kl_div:
                break

    def train_value(self, obs, returns):

        for _ in range (self.value_train_iters):
            self.value_optim.zero_grad()
    
            values = self.ac.value(obs)
            value_loss = (returns-values) ** 2
            value_loss = value_loss.mean()
    
            value_loss.backward()
            self.value_optim.step()

class ActorCriticChromosome:
    def __init__(self, state_dim, action_dim):
        # Actor-Critic neural network
        self.model = ActorCriticNN(state_dim, action_dim)

        #Hyperparameters
        self.policy_lr = random.uniform(1e-5, 1e-2)  # Expanded range
        self.value_lr = random.uniform(1e-5, 1e-2)  # Expanded range
        self.ppo_clip_val = random.uniform(0.1, 0.4)  # Expanded range
        self.target_kl_div = random.uniform(0.001, 0.05)

def create_population(pop_size, state_dim, action_dim):
    return [ActorCriticChromosome(state_dim, action_dim) for _ in range(pop_size)]

"""
When initializing PPOTrainer, you're passing the individual's hyperparameters (policy_lr, value_lr, etc.) 
to ensure the evaluation process reflects the effect of those hyperparameters.
You're sampling actions from the model and accumulating rewards to calculate the fitness score.
"""

def evaluate_population(population, env, max_steps=100):
    fitness_scores = []
    for individual in population:
        total_reward = 0

        # Initialize PPOTrainer with individual's hyperparameters
        ppo = PPOTrainer(individual.model,
                         ppo_clip_val=individual.ppo_clip_val,
                         target_kl_div=individual.target_kl_div,
                         policy_lr=individual.policy_lr,
                         value_lr=individual.value_lr)

        # Reset the environment and evaluate the model
        obs, _ = env.reset()
        for _ in range(max_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            logits, _ = individual.model(obs_tensor)  # Use the model from the individual
            act_dist = Categorical(logits=logits)
            action = act_dist.sample()
            next_obs, reward, done, _, __ = env.step(action.item())
            total_reward += reward
            obs = next_obs
            if done:
                break
        fitness_scores.append(total_reward)
    
    return fitness_scores

def select_parents(population, fitness_scores, num_parents):
   # Get indices that would sort the fitness scores in descending order
    sorted_indices = np.argsort(fitness_scores)[::-1]  # Reverse for descending order
    
    # Select the best parents based on the sorted indices
    parents = np.array(population)[sorted_indices][:num_parents]
    return parents.tolist()

def crossover(parent1, parent2):
    child = ActorCriticChromosome(state_dim, action_dim)
    for param1, param2, param_child in zip(parent1.model.parameters(), parent2.model.parameters(), child.model.parameters()):
        param_child.data = (param1.data + param2.data) / 2  # Average weights

    # Crossover for hyperparameters
    child.policy_lr = (parent1.policy_lr + parent2.policy_lr) / 2
    child.value_lr = (parent1.value_lr + parent2.value_lr) / 2
    child.ppo_clip_val = (parent1.ppo_clip_val + parent2.ppo_clip_val) / 2
    child.target_kl_div = (parent1.target_kl_div + parent2.target_kl_div) / 2
    
    return child

def mutate(individual, mutation_rate=0.01):
    for param in individual.model.parameters():
        if random.random() < mutation_rate:
            param.data += torch.randn(param.size()) * 0.1  # Add small random noise
    
    # Mutate hyperparameters
    if random.random() < mutation_rate:
        individual.policy_lr += random.uniform(-1e-5, 1e-5)
    if random.random() < mutation_rate:
        individual.value_lr += random.uniform(-1e-4, 1e-4)
    if random.random() < mutation_rate:
        individual.ppo_clip_val += random.uniform(-0.01, 0.01)
    if random.random() < mutation_rate:
        individual.target_kl_div += random.uniform(-0.001, 0.001)


def discount_rewards(rewards, gamma = 0.99):
    """
    Return discounted rewards based on the given rewards and the gamma param.
    """

    new_rewards = [float(rewards[-1])]
    for i in reversed(range(len(rewards)-1)):
        new_rewards.append(float(rewards[i]) + gamma * new_rewards[-1])
    return np.array(new_rewards[::-1])

def calculate_gaes(rewards, values, gamma = 0.99, decay = 0.97):
    """
    Return the General Advantage Estimates from the given rewards and values.
    Paper: https://arxiv.org/abs/1506.02438
    """

    next_values = np.concatenate([values[1:], [0]])
    deltas = [rew + gamma * next_val - val for rew, val, next_val in zip(rewards, values, next_values)]

    gaes = [deltas[-1]]
    for i in reversed(range(len(deltas)-1)):
        gaes.append(deltas[i] + decay * gamma * gaes[-1])

    return np.array(gaes[::-1])

def rollout(model, env, max_steps=1000):
    """
    Performs a single rollout, sampling an action and collecting data.
    Returns training data in the shape (n_steps, observation_shape) and the cumulative reward.
    """

    # Create the data storage (obs, act, reward, values, act_log_probs)
    train_data = [[], [], [], [], []]  
    obs, _ = env.reset()

    #print("Environment reset, initial observation:", obs)  # Check if env.reset() works

    ep_reward = 0
    for step in range(max_steps):

        #Convert observation to a PyTorch tensor and ensure correct shape
        obs = torch.tensor(obs, dtype=torch.float32)

        logits, val = model(obs)

        # Create action distribution and sample an action
        act_dist = Categorical(logits=logits)
        act = act_dist.sample()
        act_log_prob = act_dist.log_prob(act).item()

        act, val = act.item(), val.item()

        # Take action in the environment
        next_obs, reward, done, _, __ = env.step(act)

        # Store the data for training
        for i, item in enumerate((obs, act, reward, val, act_log_prob)):
            train_data[i].append(item)

        #print(f"Action taken: {act.item()}, Reward received: {reward}, Done: {done}")

        # Update observation and cumulative reward
        obs = next_obs
        ep_reward += reward

        # Break if the episode ends
        if done:
            #print(f"Episode ended after {step+1} steps with cumulative reward: {ep_reward}")
            break


    train_data = [np.asarray(x) for x in train_data]
    
    train_data[3] = calculate_gaes(train_data[2], train_data[3])
    
    return train_data, ep_reward

env = UnityEnvironment(file_name='../SoftRobot')

# Reset the environment
env.reset()

# Get the behavior name and the corresponding brain
behavior_name = list(env.behavior_specs)[0]
spec = env.behavior_specs[behavior_name]

# Define the environment and parameters
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
n_episodes = 200
print_freq = 20
pop_size = 30
num_parents = 10
num_generations = 50

# Initialize the population
population = create_population(pop_size, state_dim, action_dim)

def train_model():
    for generation in range(num_generations):
        # Evaluate the current population
        fitness_scores = evaluate_population(population, env)
        
        # Select the best parents
        parents = select_parents(population, fitness_scores, num_parents)
    
        next_generation = []
        
        # Print the generation number
        print("Generation: ", generation + 1)
    
        # Create a selective parents list for crossover
        selective_parents = [parents[0], parents[1], parents[2], parents[3], parents[4]]
    
        best_select = []
        
        for i in range(num_parents):
            parent1, parent2 = np.random.choice(selective_parents, 2, replace=False)
            child = crossover(parent1, parent2)
            mutate(child)
            best_select.append(child)
    
        for i in range(pop_size):
            next_generation.append(population[i])
    
        # Evaluate fitness of the new generation
        next_generation_fitness = evaluate_population(next_generation, env)
        
        # Sort the next generation based on fitness scores (best to worst)
        next_generation = [x for _, x in sorted(zip(next_generation_fitness, next_generation), key=lambda pair: pair[0], reverse=True)]
    
        index = 0
        for i in range ((pop_size-10),pop_size):
            next_generation[i] = best_select[index]
            index = index + 1
        
        population = next_generation
    
        # Optionally fine-tune the best policy with PPO after EA
        best_policy = parents[0]  # Select the best policy for further training with PPO
        ppo = PPOTrainer(best_policy.model, ppo_clip_val = best_policy.ppo_clip_val, target_kl_div=best_policy.target_kl_div, 
                         policy_lr=best_policy.policy_lr, value_lr=best_policy.value_lr)
    
        ep_rewards = []
        for episode_idx in range(n_episodes):
            # Perform rollout
            train_data, reward = rollout(best_policy.model, env)
            ep_rewards.append(reward)
    
            permute_idxs = np.random.permutation(len(train_data[0]))
            obs = torch.tensor(train_data[0][permute_idxs], dtype=torch.float32)
            act = torch.tensor(train_data[1][permute_idxs], dtype=torch.int32)
            gaes = torch.tensor(train_data[3][permute_idxs], dtype=torch.float32)
            act_log_probs = torch.tensor(train_data[4][permute_idxs], dtype=torch.float32)
    
            # Value Data
            returns = discount_rewards(train_data[2])[permute_idxs]
            returns = torch.tensor(returns, dtype=torch.float32)
    
            # Train Policy
            ppo.train_policy(obs, act, act_log_probs, gaes)
            ppo.train_value(obs, returns)
    
            # Print average reward every 'print_freq' episodes
            if (episode_idx + 1) % print_freq == 0:
                avg_reward = np.mean(ep_rewards[-print_freq:])  # Calculate the average of the last 'print_freq' rewards
                print('Generation {} | Episode {} | Avg Reward {:.1f}'.format(
                    generation + 1, episode_idx + 1, avg_reward))
    
        # Calculate and print the overall average reward for this generation
        generation_avg_reward = np.mean(ep_rewards)  # Calculate average for all episodes in this generation
        print("Generation {} Average Reward: {:.1f}".format(generation + 1, generation_avg_reward))
    
        env.reset()
        
def predict_action(observation):
    """Predicts the action given an observation."""
    obs_tensor = torch.tensor(observation, dtype=torch.float32)
    logits, _ = model(obs_tensor)  # Forward pass through the model
    act_dist = Categorical(logits=logits)
    action = act_dist.sample().item()  # Sample an action
    return action

def run_model(observation):
    """Runs the model and returns the action."""
    return predict_action(observation)

if __name__ == "__main__":
    train_model()  # Call this to start training when the script is run