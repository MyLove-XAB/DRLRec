import pandas as pd
import numpy as np
# import gym
import cfg
import torch
import time


label = np.load("../data/label_dict.npy", allow_pickle=True).item()     # 读取标签字典
items = pd.read_csv("../data/product_feature.csv")                      # 读取商品特征数据
items_raw = pd.read_csv("../data/product_raw_feature.csv")              # 读取原始商品特征，用于计算
args = cfg.get_args()                                                   # 加载超参数
feature, raw_feature = [], []
for str_values in items["feature"]:
    feature.append([float(val) for val in str_values.strip('[]').split()])

for str_values in items_raw["feature"]:
    raw_feature.append([float(val) for val in str_values.strip('[]').split()])

feature = torch.tensor(feature)
raw_feature = torch.tensor(raw_feature)


def user_ind(users, mini_batch_size):
    batch_start = np.arange(0, len(users), mini_batch_size)
    indices = np.arange(len(users), dtype=np.int64)     # 记录编号[0,1,2....19]
    np.random.shuffle(indices)                          # 打乱编号顺序[3,1,9,11....18]
    mini_batches = [indices[i:i + mini_batch_size] for i in batch_start]
    return mini_batches


class JDEnv(object):
    def __init__(self, users, user_feature):                      # users决定使用的是训练数据还是测试数据
        """

        :param users: user data
        :param user_feature: corresponding user feature
        """
        self.total_steps = 0                        # 记录总的训练数据用了多少了
        self.steps = 0
        self.batch_size = args.env_batch            # 每次选多少个user出来
        self.users_all = users  # 在训练的时候用训练数据，在测试的时候，初始化env之后，用测试数据进行替换
        self.mini_batches = user_ind(self.users_all, self.batch_size)                       # batch index list
        self.user_feature = user_feature
        self.features = feature                     # 商品特征，用于后续计算相似度
        self.index_list = list([set() for i in range(self.batch_size)])                     # 用来存储已经推荐过的商品
        self.rec_item_num = args.rec_item_num
        self.ground_truth = label                   # label_dict

    def step(self, actions):
        self.steps += 1/args.rec_runs               # 作为动态信息的一部分，表示目前推荐到第几轮了, 并做了一个归一化
        if self.steps >= 0.99:
            self.done = True

        res = torch.matmul(self.features, torch.transpose(actions, 0, 1))   # 每个商品对不同用户的评分
        reward_list = list()
        ct = 0      # click through，用来计算当前所有用户的点击率，因此不用针对每个用户进行计算

        for i in range(len(res.T)):         # 针对第i个用户计算奖励，调整state
            cur_uid = self.uid[i]           # 当前的uid
            self.reward = 0

            tmp = 0                         # 记录本轮已经推荐的商品数量
            sorted_values, sorted_indices = torch.sort(res.T[i], descending=True)       # 按分数排序，得到排序值和对应的索引

            # 计算奖励
            for id in sorted_indices[:args.rec_item_num*args.rec_runs]:     # 一个用户多轮推荐最多推荐rec_item_num*rec_runs
                sku_id = items.iloc[np.int(id)]["sku_id"]
                if sku_id not in self.index_list[i]:                        # 如果之前没有推荐过，直接对比label，计算reward
                    tmp += 1
                    self.index_list[i].add(sku_id)
                    self.id_list[i].append(id.item())
                    if sku_id in self.ground_truth[cur_uid]["level1"]:      # 下单
                        self.reward += 1.
                        ct += 1             # 用的是elif，就不用考虑一个商品多次操作的问题了，直接按照优先级进行计算
                        self.behavior[i][0] += 1
                        self.buy_list[i].append(id.item())                  # 记录购买商品id
                    elif sku_id in self.ground_truth[cur_uid]["level2"]:    # 浏览，加购物车，点击
                        self.reward += 0.5
                        ct += 1
                        self.behavior[i][1] += 1
                        # self.buy_list[i].append(id.item())                # 记录ctr商品id
                    elif sku_id in self.ground_truth[cur_uid]["level3"]:    # 关注
                        self.reward += 0.
                        self.behavior[i][2] += 1
                    elif sku_id in self.ground_truth[cur_uid]["level4"]:    # 删除购物车
                        self.reward -= 1.
                        self.behavior[i][3] += 1
                    else:                                                   # 没有点击，没有任何操作
                        self.reward -= 0.
                if tmp >= self.rec_item_num:                                # 已经推荐rec_item_num个商品，本轮结束，开启下一轮
                    break
            for tmp_i in range(4):
                self.behavior[i][tmp_i] = self.behavior[i][tmp_i]/(self.steps*10*10)

            reward_list.append(self.reward)      # 记录一轮推荐后当前用户奖励

            #  更新state
            if self.index_list[i]:          # 非空的时候更新states，并更新动态信息
                rec_raw_features = raw_feature[self.id_list[i]][:, 1:11]                # 已推荐的商品三种属性
                # 计算对应比例，[0, 1, 0, 1, ...]直接按维度sum
                dynamic1 = torch.sum(rec_raw_features, dim=0)/len(self.id_list[i])      # shape=[10]

                if self.buy_list[i]:
                    buy_raw_features = raw_feature[self.buy_list[i]][:, 1:11]           # 已推荐的商品三种属性
                    # 计算对应比例，[0, 1, 0, 1, ...]直接按维度sum
                    dynamic2 = torch.sum(buy_raw_features, dim=0) / len(self.buy_list[i])
                else:
                    dynamic2 = torch.zeros([10])         # 现在还没有购买，各种比例就是0

                # 使用索引和切片，修改动态信息, 静态动态信息拼接
                self.states[i] = torch.cat(
                    (self.states[i][:16], dynamic1, dynamic2, torch.tensor(self.behavior[i]), torch.tensor([self.steps])
                     ), dim=-1)
        return reward_list, self.states, self.done, \
               ct, ct/(10*self.batch_size)                    # 返回所有user的r, s', ct, 当前时刻的ctr

    def reset(self):            # done之后reset
        if self.total_steps >= int(len(self.users_all)/self.batch_size):             # 每遍历完一遍全部训练数据，重新打散
            self.total_steps = 0
            self.mini_batches = user_ind(self.users_all, self.batch_size)               # 重新打散
        self.steps = 0
        ind = self.mini_batches[self.total_steps]
        self.users = self.users_all.iloc[self.mini_batches[self.total_steps]]           # 打散后的用户id和特征,并选择对应轮次
        self.index_list = list([set() for i in range(self.batch_size)])     # 记录sku_id
        self.id_list = list([[] for i in range(self.batch_size)])           # 记录sku的index
        self.buy_list = list([[] for i in range(self.batch_size)])          # 记录用户购买的的index
        self.static = self.user_feature[self.mini_batches[self.total_steps]]            # 用户静态信息，转为tensor
        self.dynamic = torch.zeros([len(self.users), args.dynamic_dim])     # 动态信息初始化  # self.batch_size
        self.states = torch.cat([self.static, self.dynamic], dim=1)         # 拼接
        self.uid = self.users["user_id"].values

        self.done = False
        self.behavior = list([[0, 0, 0, 0] for i in range(self.batch_size)])    # 下单，其他ctr行为, 关注，删购物车（其余都为无行为，可以通过前4者计算出来，这里就不专门统计了）
        self.total_steps += 1

        return self.states              # 返回states


if __name__ == "__main__":
    users = pd.read_csv("../data/user_static_feature.csv")  # 读取训练数据
    user_feature = []
    for str_values in users["feature"]:
        user_feature.append([float(val) for val in str_values.strip('[]').split()])
    user_feature = torch.tensor(user_feature)
    # 环境初始化
    jdenv = JDEnv(users, user_feature)
    s = jdenv.reset()
    mean = torch.zeros([256, 32])
    std = torch.ones([256, 32])
    t1 = time.time()
    for i in range(30):
        print("step: ", i)
        actions = torch.normal(mean=mean, std=std)
        r_list, s_, d, ct, ctr = jdenv.step(actions)
        print("r:{}, ct: {}, ctr: {}".format(r_list, ct, 100*ctr))
        print(len(r_list))
    print("time: ", round(time.time()-t1, 4))

