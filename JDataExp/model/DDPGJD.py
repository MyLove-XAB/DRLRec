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
import random
import wandb
from cfg import get_args
from copy import deepcopy
from JDenv import JDEnv


device = torch.device("cuda")


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
        num_outputs = action_space

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
        num_outputs = action_space

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
    def __init__(self, gamma, tau, hidden_size, num_inputs, action_space, ae=True):

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

        return mu  # .clamp(-1, 1)  # mu在(-1, 1)之间

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
    users = pd.read_csv("../data/user_static_feature.csv")  # 读取训练数据
    user_feature = []
    for str_values in users["feature"][:int(len(users)*0.96)]:
        user_feature.append([float(val) for val in str_values.strip('[]').split()])
    user_feature = torch.tensor(user_feature)
    # read test data
    test_user_feature = []
    for str_values in users["feature"][int(len(users)*0.96):]:
        test_user_feature.append([float(val) for val in str_values.strip('[]').split()])
    test_user_feature = torch.tensor(test_user_feature)

    cfg = get_args()

    env = JDEnv(users[:int(len(users)*0.96)], user_feature)
    test_env = JDEnv(users[int(len(users)*0.96):], test_user_feature)
    agent = DDPG(gamma=0.95, tau=0.001, hidden_size=cfg.hidden_dim,
                 num_inputs=41, action_space=32)            # 41维用户特征，32维商品特征

    memory = ReplayMemory(10000)

    ounoise = OUNoise(32)           # 动作维度
    param_noise = None

    test_result = []

    for i_episode in tqdm(range(int(cfg.ddpgtrain_eps))):       # 训练一定次数, 每轮针对env_batch个用户推荐
        state = env.reset()                 # (env_batch, 41)41维: 16+25(4+10+10+1)
        episode_reward = 0
        sum_ctr = 0
        steps = 0
        while True:     # sequential
            action = agent.select_action(state, ounoise, param_noise).cpu()     # 32维,
            reward_list, next_state, done, ct, ctr = env.step(action)
            episode_reward += sum(reward_list)/len(reward_list)                 # 平均一下，避免最后一组用户数量少的影响
            steps += 1
            sum_ctr += ctr

            mask = torch.Tensor([not done])         # 记录是否完成当前用户的全部推荐

            for i in range(len(reward_list)):       # save batch experience
                memory.push(state[i].reshape([-1, 41]), action[i].reshape([-1, 32]), mask,
                            next_state[i].reshape([-1, 41]), torch.tensor([reward_list[i]]))
                # 每次存入的都是不同用户的推荐轨迹，能够在一定程度上降低数据间的相关性

            state = next_state
            if len(memory) >= 2000:
                transitions = memory.sample(cfg.ddpgbatch_size)     # batch_size
                batch = Transition(*zip(*transitions))              # tuple存储trajectories
                value_loss, policy_loss = agent.update_parameters(batch)
            if done:
                break

        if i_episode % 2 == 0:   # 每间隔一定轮数，测一次测试集表现，注意此时需要加载测试集重置环境，并在最后将环境重新加载维训练集，或者直接新建一个环境（起个不同的名字）
            test_episode_reward = 0  # 记录全部测试数据的，而不单是某一轮的
            test_sum_ctr = 0
            t_steps = 0
            for test_i_episode in tqdm(range(int(len(test_user_feature)//cfg.env_batch + 1))):  # 遍历全部测试集
                state = test_env.reset()        # (env_batch, 41)41维: 16+25(4+10+10+1)
                while True:                     # sequential
                    action = agent.select_action(state, ounoise, param_noise).cpu()     # 32维,
                    reward_list, next_state, done, ct, ctr = test_env.step(action)
                    test_episode_reward += sum(reward_list) / len(reward_list)  # 平均一下，避免最后一组用户数量少的影响
                    t_steps += 1
                    test_sum_ctr += ctr
                    state = next_state

                    if done:
                        break
            test_result.append([i_episode, test_episode_reward, 100*test_sum_ctr / t_steps])

    # save test result
    test_result_array = np.array(test_result)
    test_result_df = pd.DataFrame()
    test_result_df["episode_step"] = test_result_array[:, 0]
    test_result_df["avg_reward"] = test_result_array[:, 1]
    test_result_df["ctr"] = test_result_array[:, 2]

    test_result_df.to_csv("./new_result/ddpg_base.csv")
