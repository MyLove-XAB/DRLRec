import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import torch
import torch.nn as nn
from virtualTB.model.ActionModel import ActionModel
from virtualTB.model.LeaveModel import LeaveModel
from virtualTB.model.UserModel import UserModel
from virtualTB.utils import *


class VirtualTB(gym.Env):  # 继承gym.Env，然后可以用gym.make进行调用
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.n_item = 5
        self.n_user_feature = 88
        self.n_item_feature = 27
        self.max_c = 100
        self.obs_low = np.concatenate(([0] * self.n_user_feature, [0,0,0]))
        self.obs_high = np.concatenate(([1] * self.n_user_feature, [29,9,100]))
        self.observation_space = spaces.Box(low = self.obs_low, high = self.obs_high, dtype = np.int32)
        self.action_space = spaces.Box(low = -1, high = 1, shape = (self.n_item_feature,), dtype = np.float32)
        self.user_model = UserModel()
        self.user_model.load()
        self.user_action_model = ActionModel()  # 用来根据动作进行预测
        self.user_action_model.load()
        self.user_leave_model = LeaveModel()
        self.user_leave_model.load()
        self.reset()

    def seed(self, sd=0):
        torch.manual_seed(sd)

    @property
    def state(self):  # 返回当前状态，在step()函数中，返回动作执行之后的state
        return np.concatenate((self.cur_user, self.lst_action, np.array([self.total_c])), axis=-1)

    def __user_generator(self):
        # with shape(n_user_feature,)
        user = self.user_model.generate()
        self.__leave = self.user_leave_model.predict(user)
        return user

    def step(self, action):
        # Action: tensor with shape (27, )
        self.lst_action = self.user_action_model.predict(FLOAT(self.cur_user).unsqueeze(0),
                                                         FLOAT([[self.total_c]]),
                                                         FLOAT(action).unsqueeze(0)
                                                         ).detach().numpy()[0]
        reward = int(self.lst_action[0])  # 1维
        self.total_a += reward
        self.total_c += 1
        self.rend_action = deepcopy(self.lst_action)
        done = (self.total_c >= self.__leave)  # 判断user是否离开
        if done:
            self.cur_user = self.__user_generator().squeeze().detach().numpy()
            self.lst_action = FLOAT([0, 0])  # 重置
        return self.state, reward, done, {'CTR': self.total_a / self.total_c / 10}

    def reset(self):
        self.total_a = 0
        self.total_c = 0
        self.cur_user = self.__user_generator().squeeze().detach().numpy()
        self.lst_action = FLOAT([0, 0])
        self.rend_action = deepcopy(self.lst_action)
        return self.state

    def render(self, mode='human', close=False):
        print('Current State:')
        print('\t', self.state)
        a, b = np.clip(self.rend_action, a_min = 0, a_max = None)
        print('User\'s action:')
        print('\tclick:%2d, leave:%s, index:%2d' % (int(a), 'True' if self.total_c > self.max_c else 'False', int(self.total_c)))
        print('Total clicks:', self.total_a)
