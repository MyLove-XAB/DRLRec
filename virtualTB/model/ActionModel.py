import os
from virtualTB.utils import *


class ActionModel(nn.Module):
    def __init__(self, n_input=88 + 1 + 27, n_output=11 + 10, learning_rate=0.01):
        super(ActionModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_input, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, n_output)
        )
        self.max_a = 11
        self.max_b = 10

    def predict(self, user, page, weight):
        x = self.model(torch.cat((user, page, weight), dim=-1))
        a = torch.multinomial(F.softmax(x[:, :self.max_a], dim=1), 1)  # reward, 返回的是下标 0~10，即这一页点击的数量？可能是提前train好的原因吧
        # if a == 10:
        #     print("index")
        #     print(10)
        b = torch.multinomial(F.softmax(x[:, self.max_a:], dim=1), 1)
        return torch.cat((a, b), dim=-1)

    def load(self, path=None):
        if path == None:
            path = os.path.dirname(__file__) + '/../data/action_model.pt'  # 加载训练好的action_model
        self.model.load_state_dict(torch.load(path))
