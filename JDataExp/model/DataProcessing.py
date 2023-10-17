import pandas as pd
import numpy as np
import os
import re
from tqdm import tqdm


# read data
data_list = os.listdir("../data")
ac1 = pd.read_csv("../data/JData_Action_201602.csv")
ac2 = pd.read_csv("../data/JData_Action_201603.csv")
ac3 = pd.read_csv("../data/JData_Action_201604.csv")
usr = pd.read_csv("../data/JData_User.csv")

# action cate == 8
action_df = pd.concat([ac1, ac2, ac3], axis=0)
action_new = action_df[action_df["cate"]==8]        # 18128055
action_new.to_csv("../data/JData_cate8.csv")

# ground truth
action_new.drop(["model_id"], axis=1, inplace=True)
ground_truth_dict = dict()
for i in tqdm(range(len(action_new))):
    if action_new["user_id"][i] not in ground_truth_dict:
        ground_truth_dict[action_new["user_id"][i]] = set()
        ground_truth_dict[action_new["user_id"][i]].add(action_new["sku_id"][i])

    else:
        ground_truth_dict[action_new["user_id"][i]].add(action_new["sku_id"][i])

# Select user and product that in cate8
usr_new = usr[usr["user_id"].isin(ground_truth_dict.keys())]        # 104740
usr_new.to_csv("../data/User_new.csv")
# product are all in cate8                  # 24187
cate8 = pd.read_csv("../data/JData_cate8.csv")
print(len(ground_truth_dict.keys()))        # 104740
print(len(set(cate8["sku_id"].values)))     # 3938

# sample
usr = pd.read_csv("../data/User_new.csv")
usr_sample = usr.sample(frac=0.2, random_state=42)
usr_sample_set = set(usr_sample["user_id"].values)          # 20948
cate_sample = cate8[cate8["user_id"].isin(usr_sample_set)]
print(len(cate_sample))         # 3597713
print(len(set(cate_sample["sku_id"].values)))               # 3329
usr_sample.to_csv("../data/User_sample.csv")
cate_sample.to_csv("../data/Cate_sample.csv")

# item features
product = pd.read_csv("../data/JData_Product.csv", encoding="gbk")
comment = pd.read_csv("../data/JData_Comment.csv", encoding="gbk")

product_sample = product[product["cate"]==8]
comment_sample = comment[comment["cate"]==8]
product_result = pd.merge(product_sample, comment_sample, on='sku_id', how='outer')
product_unique = product_result.drop_duplicates(subset="sku_id")
product_unique.drop(["dt"], axis=1, inplace=True)           # 24187
product_final = product_unique.fillna(0.)                   # 缺失值填充
product_final.drop(["cate"], axis=1, inplace=True)
product_final.to_csv("../data/product_final.csv")

product = pd.read_csv("../data/product_final.csv")
# 归一化
product["comment_num"] = product["comment_num"]/product["comment_num"].max()
product["has_bad_comment"] = product["has_bad_comment"]/product["has_bad_comment"].max()

# one-hot编码
attr1_df = pd.get_dummies(product["a1"], prefix="a1")
attr2_df = pd.get_dummies(product["a2"], prefix="a2")
attr3_df = pd.get_dummies(product["a3"], prefix="a3")
attr4_df = pd.get_dummies(product["brand"], prefix="brand")
product_feature = pd.concat([product[['sku_id', 'bad_comment_rate']], attr1_df, attr2_df, attr3_df, attr4_df], axis=1)

# raw item feature
selected_list = product_feature.columns[1:]
feature_vector = product_feature[selected_list].values
df = pd.DataFrame()
df["sku_id"] = product["sku_id"]
df["feature"] = list(feature_vector)                    # 113维
df.to_csv("../data/product_raw_feature.csv")            # sku_id, feature ndarray, 已经shuffle过了

# user id, 对应的label
cate_sample = pd.read_csv("../data/Cate_sample.csv")
cate_sample.drop(["Unnamed: 0"], axis=1, inplace=True)
cate_sample.drop(["Unnamed: 0.1"], axis=1, inplace=True)
cate_sample.drop(["time", "model_id", "cate", "brand"], axis=1, inplace=True)   # type是用户行为，需要保留

label_dict = dict()
l1 = {4}            # 下单
l2 = {1, 2, 6}      # 浏览加购物车，点击
l3 = {5}            # 关注
l4 = {3}            # 删除购物车
for id in tqdm(range(len(cate_sample))):
    uid = cate_sample.iloc[id]["user_id"]
    utype = cate_sample.iloc[id]["type"]
    item = cate_sample.iloc[id]["sku_id"]
    if uid not in label_dict:
        label_dict[uid] = dict({"level1": set(), "level2": set(), "level3": set(), "level4": set()})
    if utype in l1:
        label_dict[uid]["level1"].add(item)
    elif utype in l2:
        label_dict[uid]["level2"].add(item)
    elif utype in l3:
        label_dict[uid]["level3"].add(item)
    else:
        label_dict[uid]["level4"].add(item)

# save
np.save("../data/label_dict.npy", label_dict)
label_dict= np.load("../data/label_dict.npy", allow_pickle=True).item()

usr_sample = pd.read_csv("../data/User_sample.csv")
usr_sample.drop(["Unnamed: 0", "Unnamed: 0.1"], axis=1, inplace=True)


def convert_age(age_str):
    if age_str == u'-1':
        return 0
    elif age_str == u'15岁以下':
        return 1
    elif age_str == u'16-25岁':
        return 2
    elif age_str == u'26-35岁':
        return 3
    elif age_str == u'36-45岁':
        return 4
    elif age_str == u'46-55岁':
        return 5
    elif age_str == u'56岁以上':
        return 6
    else:
        return -1


usr_sample['age'] = usr_sample['age'].map(convert_age)
age_df = pd.get_dummies(usr_sample["age"], prefix="age")
sex_df = pd.get_dummies(usr_sample["sex"], prefix="sex")
user_lv_df = pd.get_dummies(usr_sample["user_lv_cd"], prefix="user_lv_cd")
user_feature = pd.concat([usr_sample['user_id'], age_df, sex_df, user_lv_df], axis=1)
selected_list2 = user_feature.columns[1:]
feature_vector = user_feature[selected_list2].values

df2 = pd.DataFrame()
df2["user_id"] = user_feature["user_id"]
df2["feature"] = list(feature_vector)           # 16维
df2.to_csv("../data/user_static_feature.csv")   # 使用的时候还需要shuffle一下
