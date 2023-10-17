import numpy as np


dataset = []
# 用dataset.txt里面的数据训练AE
with open("../SupervisedLearning/dataset.txt") as f:
    for data in f.readlines():
        tmp = data.split("\t")
        state = np.array([np.float(x) for x in tmp[0].split(",")])
        dataset.append(state)  # 91

length = len(dataset)
dataset = np.array(dataset).reshape([length, -1])
static_state = dataset[:, :88]
dynamic_state = dataset[:, 88:91]

alldata = np.unique(static_state, axis=0)  # 动态特征初始化的时候都是0，
print("all data length: ", len(alldata))
np.random.seed(1)
index = np.random.randint(0, len(alldata), size=1000)
testdata = alldata[index]
testdata = np.unique(testdata, axis=0)
print("test data length: ", len(testdata))
np.save("./new_result/testdata.npy", testdata)
