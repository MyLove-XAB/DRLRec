import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from cfg import get_args
from tqdm import tqdm


class AutoEncoder(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output) -> None:
        super().__init__()
        self.fc1 = nn.Linear(n_feature, n_hidden)  # 设置第一个全连接层(输入层到隐藏层): 状态数个神经元到隐藏层数目个神经元
        self.BN = nn.LayerNorm(normalized_shape=n_hidden)
        self.out = nn.Linear(n_hidden, n_output)  # 88 --> 128 --> 64

        self.decoder = nn.Sequential(
            nn.Linear(n_output, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_feature),  # 32 --> 64 --> 88
        )

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(self.BN(x))
        x = self.out(x)
        x = self.decoder(x)
        return x


product = pd.read_csv("../data/product_raw_feature.csv")
features = np.array(product["feature"])
dataset = []
for str_values in features:
    dataset.append([float(val) for val in str_values.strip('[]').split()])

length = len(dataset)
dataset = np.array(dataset).reshape([length, -1])

np.random.seed(1)


def train(epoch=10):
    data_train = dataset[:int(len(dataset) * 0.9)]
    data_test = dataset[int(len(dataset) * 0.9):]
    data_train = torch.Tensor(data_train)
    data_test = torch.Tensor(data_test)
    train_loader = DataLoader(data_train, batch_size=256, shuffle=True)
    test_loader = DataLoader(data_test, batch_size=256, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AutoEncoder(n_feature=113, n_hidden=64, n_output=32).to(device)
    criteon = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epoch):
        for batch_idx, x in enumerate(train_loader):
            x = x.to(device)
            x_hat = model(x)
            loss = criteon(x_hat, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(epoch, 'loss:', loss.item())
        if (epoch + 1) % 10 == 0:
            for batch_idx, x, in enumerate(test_loader):
                x = x.to(device)
                with torch.no_grad():
                    x_hat = model(x)
                    loss = criteon(x_hat, x)
            print('{}epochs, test_loss:{}'.format(epoch, loss.item()))

    pre_dict = model.state_dict()  # 按键值对将模型参数加载到pre_dict
    # print(pre_dict)
    print(pre_dict.keys())

    torch.save(pre_dict, "./product_AE.pkl")

    # process
    res = []
    for batch_idx in tqdm(range(int(len(dataset)/256 + 1))):
        x = torch.tensor(dataset[batch_idx*256: (batch_idx+1)*256], dtype=torch.float32)
        x = x.to(device)
        x1 = model.fc1(x)
        x2 = model.BN(x1)
        x_hat = model.out(x2)
        for tmp in x_hat:
            res.append(np.array(tmp.cpu().detach().numpy()))

    df = pd.DataFrame()
    df["sku_id"] = product["sku_id"]
    df["feature"] = res
    df.to_csv("../data/product_feature.csv")  # str


if __name__ == "__main__":
    train(epoch=100)


