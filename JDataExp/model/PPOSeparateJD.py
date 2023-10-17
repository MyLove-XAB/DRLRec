import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
import wandb
import pandas as pd
from cfg import get_args
from tqdm import tqdm
from copy import deepcopy
import random
from JDenv import JDEnv


class PPOmemory:
    def __init__(self, mini_batch_size):
        self.states = []    # 状态
        self.actions = []   # 实际采取的动作
        self.probs = []     # 动作概率
        self.vals = []      # critic输出的状态值
        self.rewards = []   # 奖励
        self.dones = []     # 结束标志

        self.mini_batch_size = mini_batch_size  # minibatch的大小

    def sample(self):
        n_states = len(self.states)  # memory记录数量=20
        batch_start = np.arange(0, n_states, self.mini_batch_size)  # 每个batch开始的位置[0,5,10,15]
        indices = np.arange(n_states, dtype=np.int64)  # 记录编号[0,1,2....19]
        np.random.shuffle(indices)  # 打乱编号顺序[3,1,9,11....18]
        mini_batches = [indices[i:i + self.mini_batch_size] for i in batch_start]
        # 生成4个minibatch，每个minibatch记录乱序且不重复，用于后续学习更新网络

        return np.array([np.array(x) for x in self.states]), \
               torch.tensor(np.array([tensor.cpu().numpy() for tensor in self.actions])), \
               np.array(self.probs),  \
               np.array(self.vals), np.array(self.rewards), np.array(self.dones), mini_batches
        # [tensor.detach().numpy() for tensor in self.probs]

    # 每一步都存储trace到memory
    def push(self, state, action, prob, val, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.vals.append(val)
        self.rewards.append(reward)
        self.dones.append(done)

    # 固定步长更新完网络后清空memory
    def clear(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []


# actor:policy network
class Actor(nn.Module):
    def __init__(self, n_states, n_actions, cfg):
        super(Actor, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(n_states-25, cfg.hidden_dim),
            nn.Tanh(),
            nn.Linear(cfg.hidden_dim, cfg.out_dim),
            nn.Tanh(),
        )

        self.linear = nn.Linear(25, 32)
        self.actor1 = nn.Sequential(nn.Linear(cfg.out_dim, n_actions),
                                    nn.Tanh())  # V(static)
        self.actor2 = nn.Sequential(nn.Linear(cfg.out_dim+32, n_actions),
                                    nn.Tanh())  # V(importance)
        self.mu = nn.Linear(n_actions, n_actions)
        self.sigma = nn.Linear(cfg.out_dim+32, n_actions)

    def forward(self, state):
        x = state[:, :16]
        ae_state = self.actor(x)
        x1 = self.actor1(ae_state)

        y = state[:, 16:]
        y = self.linear(y)
        y = F.relu(y)
        new_feature = torch.cat((ae_state, y), dim=1)

        x2 = self.actor2(new_feature)
        mu = torch.add(x1, x2)/2  # [-2, 2]

        sigma = F.softplus(self.sigma(new_feature)) + 0.001
        return mu, sigma
        #  Creates a categorical distribution parameterized by either :attr:`probs` or:attr:`logits` (but not both).


# critic:value network
class Critic(nn.Module):
    def __init__(self, n_states, hidden_dim):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(n_states, hidden_dim),  # 91
            nn.Tanh(),
            nn.Linear(hidden_dim, 1))

    def forward(self, state):
        value = self.critic(state)
        return value


class Agent:
    def __init__(self, n_states, n_actions, cfg, ae=True):
        # 训练参数
        self.gamma = cfg.gamma  # 折扣因子
        self.n_epochs = cfg.n_epochs  # 每次更新重复次数
        self.gae_lambda = cfg.gae_lambda  # GAE参数
        self.policy_clip = cfg.policy_clip  # clip参数
        # self.device = torch.device(cfg.device)  # 运行设备  torch.device(cfg.device)返回的就是str "cuda"
        self.device = cfg.device

        # AC网络及优化器
        self.actor = Actor(n_states, n_actions, cfg)
        self.critic = Critic(n_states, cfg.hidden_dim)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        self.actor.to(self.device)
        self.critic.to(self.device)

        # 经验池
        self.memory = PPOmemory(cfg.mini_batch_size)

    def choose_action(self, state):
        self.actor.eval()
        state = state.to(self.device)  # 数组变成张量
        mu, sigma = self.actor(state)  # action分布
        value = self.critic(state)  # state value值
        try:
            dist = torch.distributions.normal.Normal(mu, sigma)

        except:
            print(mu)
            # mu, sigma = self.actor(state)
            print(sigma)
            mu = torch.zeros(mu.shape)
            sigma = F.softplus(torch.full(sigma.shape, fill_value=0.1)) + 0.001
            dist = torch.distributions.normal.Normal(mu, sigma)
        action = dist.sample()  # 根据概率选择action

        prob = dist.log_prob(action).sum(axis=-1)

        return action, prob, value

    def greedy_action(self, state):
        state = torch.tensor(state, dtype=torch.float).to(self.device)  # 数组变成张量
        _, dist_ = self.actor(state)  # action分布
        action = torch.squeeze(torch.argmax(dist_)).item()  # greedy selection
        prob = torch.squeeze(dist_[action]).item()
        value = self.critic(state)  # state value值
        value = torch.squeeze(value).item()
        return action.clamp(-1, 1), prob, value

    def learn(self):
        for _ in range(self.n_epochs):  # 决定经验的重复利用次数，每一次都遍历一遍经验，但是sample的顺序不同
            # memory中的trace以及处理后的mini_batches，mini_batches只是trace索引而非真正的数据
            states_arr, actions_arr, old_probs_arr, vals_arr,\
                rewards_arr, dones_arr, mini_batches = self.memory.sample()

            # 计算GAE
            values = vals_arr[:]  # ndarray, shape: (batch_size,)和vals_arr一样 没必要重复赋值
            advantage = np.zeros(len(rewards_arr), dtype=np.float32)
            for t in range(len(rewards_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(rewards_arr) - 1):
                    a_t += discount * (rewards_arr[k] + self.gamma * values[k+1] * (1 - int(dones_arr[k])) - values[k])
                    # 第一个时间点不用乘lambda, 所以discount是1，从第二个时间点开始不断地累乘lambda
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t
            advantage = torch.tensor(advantage).to(self.device)  # # ndarray, shape: (batch_size,)

            # mini batch 更新网络
            values = torch.tensor(values).to(self.device)
            for batch in mini_batches:  # mini_batches: [batch_size/mini_batch_size, mini_batch_size]
                states = torch.tensor(states_arr[batch], dtype=torch.float).to(self.device)
                # shape: [mini_batch_size, n_states]
                old_probs = torch.tensor([x.data for x in old_probs_arr[batch]])

                ac_list = [tensor.cpu().numpy() for tensor in actions_arr[batch]]
                actions = torch.tensor(ac_list).to(self.device)  # shape: [mini_batch_size, 1]

                # mini batch 更新一次critic和actor的网络参数就会变化
                # 需要重新计算新的dist, values, probs得到ratio,即重要性采样中的新旧策略比值, 在没更新前二者的probs是一样的
                mu, sigma = self.actor(states)  # [mini_batch_size, n_actions]

                critic_value = torch.squeeze(self.critic(states))  # shape: [mini_batch_size, 1]
                dist = torch.distributions.normal.Normal(mu.reshape([len(batch), -1]), sigma.reshape([len(batch), -1]))

                new_probs = dist.log_prob(actions).sum(axis=-1).cpu()  #

                prob_ratio = new_probs.exp() / old_probs.exp()  # shape: [mini_batch_size, 1]

                weighted_probs = advantage[batch] * prob_ratio  # shape: [mini_batch_size, 1]
                weighted_clip_probs = torch.clamp(
                    prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantage[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clip_probs).mean()

                # critic loss
                returns = advantage[batch] + values[batch]  # 根据advantage 的计算公式
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()  # tensor(3.9653e+13, dtype=torch.float64, grad_fn=<MeanBackward0>)
                # total_loss
                total_loss = actor_loss + 0.5 * critic_loss
                # 1/2便于求导，tensor(1.9827e+13, dtype=torch.float64, grad_fn=<PermuteBackward>)

                # 更新
                self.actor_optim.zero_grad()
                self.critic_optim.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.actor_optim.step()
                self.critic_optim.step()
        self.memory.clear()  # 经验遍历过n_epoch之后，用完之后立即清空，后面重新进行交互采样填充


if __name__ == "__main__":
    users = pd.read_csv("../data/user_static_feature.csv")  # 读取训练数据
    user_feature = []
    for str_values in users["feature"][:int(len(users) * 0.96)]:
        user_feature.append([float(val) for val in str_values.strip('[]').split()])
    user_feature = torch.tensor(user_feature)
    # read test data
    test_user_feature = []
    for str_values in users["feature"][int(len(users) * 0.96):]:
        test_user_feature.append([float(val) for val in str_values.strip('[]').split()])
    test_user_feature = torch.tensor(test_user_feature)

    cfg = get_args()
    cfg.device = "cpu"

    env = JDEnv(users[:int(len(users) * 0.96)], user_feature)               # 训练用数据
    test_env = JDEnv(users[int(len(users) * 0.96):], test_user_feature)     # 测试用数据

    agent = Agent(41, 32, cfg)
    # cfg.device = "cuda"
    # env.to(cfg.device)
    test_result = []

    for i_ep in tqdm(range(cfg.train_eps)):
        state = env.reset()
        episode_reward = 0
        sum_ctr = 0
        steps = 0
        done = False
        while not done:
            action, prob, val = agent.choose_action(state)
            reward_list, next_state, done, ct, ctr = env.step(action.cpu())
            steps += 1
            episode_reward += sum(reward_list)/len(reward_list)
            sum_ctr += ctr
            for i in range(len(reward_list)):

                agent.memory.push(state[i], action[i].to(cfg.device), prob[i],
                                  val[i].item(), reward_list[i], torch.tensor(done))

            agent.learn()
            state = next_state

        if i_ep % 4 == 0:
            test_ep_reward = 0
            test_ep_step = 0
            test_sum_ctr = 0
            for i in tqdm(range(int(len(test_user_feature)//cfg.env_batch + 1))):
                t_state = test_env.reset()
                done = False
                while not done:
                    action, prob, val = agent.choose_action(t_state)
                    reward_list, t_next_state, done, ct, ctr = test_env.step(action.cpu())
                    test_ep_step += 1
                    test_ep_reward += sum(reward_list)/len(reward_list)
                    state = t_next_state
                    test_sum_ctr += ctr

            test_result.append([i_ep, test_ep_reward, test_sum_ctr / steps])
    print('完成训练！')

    test_result_array = np.array(test_result)
    test_result_df = pd.DataFrame()
    test_result_df["episode_step"] = test_result_array[:, 0]
    test_result_df["avg_reward"] = test_result_array[:, 1]
    test_result_df["ctr"] = test_result_array[:, 2]

    test_result_df.to_csv("./new_result/ppo_separate.csv")

