import sys
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F
from collections import namedtuple
from tqdm import tqdm
import pandas as pd
import numpy as np
import gym
import random
import virtualTB
import wandb
from cfg import get_args


device = torch.device("cuda")  # cpu or cuda, cuda: 100回合1:42, cpu: 100回合 1:55


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)

        y = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            y = self.gamma.view(*shape) * y + self.beta.view(*shape)
        return y


nn.LayerNorm = LayerNorm


class Actor(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Actor, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        self.linear1 = nn.Linear(num_inputs, hidden_size)

        self.linear2 = nn.Linear(hidden_size, cfg.out_dim)

        self.mu = nn.Linear(cfg.out_dim, num_outputs)
        self.mu.weight.data.mul_(0.1)
        self.mu.bias.data.mul_(0.1)

    def forward(self, inputs):
        x = inputs.to(device)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        mu = torch.tanh(self.mu(x))  # (-1, 1)
        return mu


class Critic(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Critic, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)

        self.linear2 = nn.Linear(hidden_size + num_outputs, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

        self.V = nn.Linear(hidden_size, 1)
        self.V.weight.data.mul_(0.1)
        self.V.bias.data.mul_(0.1)

    def forward(self, inputs, actions):
        x = inputs.to(device)
        x = self.linear1(x)
        x = self.ln1(x)
        x = F.relu(x)

        x = torch.cat((x, actions.to(device)), 1)
        x = self.linear2(x)
        x = self.ln2(x)
        x = F.relu(x)
        V = self.V(x)
        return V


class DDPG(object):
    def __init__(self, gamma, tau, hidden_size, num_inputs, action_space):

        self.num_inputs = num_inputs
        self.action_space = action_space

        self.actor = Actor(hidden_size, self.num_inputs, self.action_space)
        self.actor_target = Actor(hidden_size, self.num_inputs, self.action_space)
        self.actor_perturbed = Actor(hidden_size, self.num_inputs, self.action_space)
        self.actor_optim = Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(hidden_size, self.num_inputs, self.action_space)
        self.critic_target = Critic(hidden_size, self.num_inputs, self.action_space)
        self.critic_optim = Adam(self.critic.parameters(), lr=1e-3)

        self.gamma = gamma
        self.tau = tau

        self.actor.to(device)
        self.actor_target.to(device)
        self.actor_perturbed.to(device)

        self.critic.to(device)
        self.critic_target.to(device)

        hard_update(self.actor_target, self.actor)  # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)

    def select_action(self, state, action_noise=None, param_noise=None):
        self.actor.eval()
        if param_noise is not None:
            mu = self.actor_perturbed((Variable(state)))
        else:
            mu = self.actor((Variable(state)))  # actor计算得到动作（actor的结构可以调整）

        self.actor.train()
        mu = mu.data

        if action_noise is not None:
            mu += torch.Tensor(action_noise.noise()).to(device)

        return mu  # mu在(-1, 1)之间

    def update_parameters(self, batch):
        state_batch = Variable(torch.cat(batch.state))
        action_batch = Variable(torch.cat(batch.action))
        reward_batch = Variable(torch.cat(batch.reward))
        mask_batch = Variable(torch.cat(batch.mask))
        next_state_batch = Variable(torch.cat(batch.next_state))

        next_action_batch = self.actor_target(next_state_batch)
        next_state_action_values = self.critic_target(next_state_batch, next_action_batch)

        reward_batch = reward_batch.unsqueeze(1).to(device)
        mask_batch = mask_batch.unsqueeze(1).to(device)
        expected_state_action_batch = reward_batch + (self.gamma * mask_batch * next_state_action_values)

        self.critic_optim.zero_grad()

        state_action_batch = self.critic((state_batch), (action_batch))

        value_loss = F.mse_loss(state_action_batch, expected_state_action_batch)
        value_loss.backward()
        self.critic_optim.step()

        self.actor_optim.zero_grad()

        policy_loss = -self.critic((state_batch), self.actor((state_batch)))

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return value_loss.item(), policy_loss.item()

    def perturb_actor_parameters(self, param_noise):
        """Apply parameter noise to actor model, for exploration"""
        hard_update(self.actor_perturbed, self.actor)
        params = self.actor_perturbed.state_dict()
        for name in params:
            if 'ln' in name:
                pass
            param = params[name]
            param += torch.randn(param.shape) * param_noise.current_stddev

    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/ddpg_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "models/ddpg_critic_{}_{}".format(env_name, suffix)
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.actor.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))


FLOAT = torch.FloatTensor
LONG = torch.LongTensor

Transition = namedtuple('Transition',
                        ('state', 'action', 'mask', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class OUNoise:

    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale


if __name__ == "__main__":
    os.environ["WANDB_API_KEY"] = 'key'
    wandb.init(name='DDPGbase', project="DRLRecVTB")
    cfg = get_args()
    env = gym.make('VirtualTB-v0')

    # env.seed(0)
    # np.random.seed(0)
    # torch.manual_seed(0)

    test_data = np.load("./new_result/testdata.npy")

    agent = DDPG(gamma=0.95, tau=0.001, hidden_size=128,
                 num_inputs=env.observation_space.shape[0], action_space=env.action_space)

    memory = ReplayMemory(100000)

    ounoise = OUNoise(env.action_space.shape[0])
    param_noise = None

    rewards = []
    total_numsteps = 0
    updates = 0
    test_result = []

    steps = 0

    for i_episode in tqdm(range(int(cfg.train_eps / 5))):
        state = torch.Tensor([env.reset()])  # 91维: 88 + [0, 0] + total_c
        episode_step = 0
        episode_reward = 0
        while True:  # sequential
            action = agent.select_action(state, ounoise, param_noise).cpu()  # 27维
            next_state, reward, done, _ = env.step(action.numpy()[0])
            total_numsteps += 1
            episode_reward += reward
            episode_step += 1
            steps += 1

            action = torch.Tensor(action)
            mask = torch.Tensor([not done])
            next_state = torch.Tensor([next_state])
            reward = torch.Tensor([reward])

            memory.push(state, action, mask, next_state, reward)

            state = next_state

            if steps % cfg.batch_size == 0:
                for _ in range(1):
                    transitions = memory.sample(cfg.batch_size)  # batch_size
                    batch = Transition(*zip(*transitions))  # tuple存储trajectories

                    value_loss, policy_loss = agent.update_parameters(batch)

                    updates += 1
            if done:
                break
        wandb.log({"train episode reward": episode_reward})
        wandb.log({"train ctr": episode_reward / episode_step / 10})

        rewards.append(episode_reward)
        if i_episode % (int(cfg.test_freq / 4)) == 0:
            episode_reward = 0
            episode_step = 0
            for i in range(len(test_data)):
                env.reset()
                env.cur_user = test_data[i][:88]
                state = torch.Tensor([env.state])
                while True:
                    action = agent.select_action(state).cpu()

                    next_state, reward, done, info = env.step(action.numpy()[0])
                    episode_reward += reward
                    episode_step += 1

                    next_state = torch.Tensor([next_state])

                    state = next_state
                    if done:
                        break

            wandb.log({"test avg episode reward": round(episode_reward / len(test_data), 2)})
            wandb.log({"test avg ctr": round(episode_reward / episode_step / 10, 4)})
            print("Episode: {}, total numsteps: {}, average reward: {}, CTR: {}".format(
                i_episode, episode_step, episode_reward / len(test_data),
                                         episode_reward / episode_step / 10))  # 每页显示10个item
            test_result.append([episode_step, episode_reward / len(test_data), episode_reward / episode_step / 10])
            # (total numsteps, average reward, ctr)

    env.close()

    # save test result
    test_result_array = np.array(test_result)
    test_result_df = pd.DataFrame()
    test_result_df["episode_step"] = test_result_array[:, 0]
    test_result_df["avg_reward"] = test_result_array[:, 1]
    test_result_df["ctr"] = test_result_array[:, 2]

    test_result_df.to_csv("./new_result/ddpg_baseline.csv")
