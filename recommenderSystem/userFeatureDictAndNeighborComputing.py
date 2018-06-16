# -*- coding:utf-8 -*-
#!/usr/local/bin/python3
# Obtain user feature dictionary -- {user_id:[]}
import pandas as pd
import os
import numpy as np
import math
import csv
from sklearn import preprocessing
data_dir = '/Users/wuyanxue/Documents/GitHub/hands-on-ml-with-sklearn-and-TF/data/recommenderData/IJCAI16_shop_user_data'
os.chdir(data_dir)


def sim(user_features, seller_features):
    '''
    Compute the similarity between user and seller
    '''
    user_buy = np.array(user_features[1::2])
    user_click = np.array(user_features[::2])
    seller_prod = np.array(seller_features)
    sim = sum(user_buy[seller_prod == 1]) / (sum(user_buy) + 0.1) + \
        0.1 * sum(user_click[seller_prod == 1]) / (sum(user_click) + 0.1)
    return sim


recommend_user_num = 100

f = open("feature.csv")
context = f.readlines()
u_feature = {}
for line in context:
    line = line.replace('\n', '')
    array = line.split(',')
    if array[0] in u_feature:
        i = 2*int(array[1])+int(array[2])-2
        u_feature[array[0]][i] = int(array[3])
    else:
        u_feature[array[0]] = [0 for i in range(144)]

# Count the sellers and its products
df = pd.read_csv('taobao.csv',header=None)

sellers_set = set(df.values[:, 1])
sellers_set_with_catogories = {}

# Construct seller features
for seller in sellers_set:
    sellers_set_with_catogories[seller] = np.zeros(72)  # 72 catogories
    t = np.array(list(set(df.values[df.values[:, 1] == seller, 3])))
    sellers_set_with_catogories[seller][t - 1] = 1

seller_recommend_user_list = {}

# Record user idx
users = np.zeros(len(u_feature.keys()), dtype=int)
for i, user in enumerate(u_feature.keys()):
    users[i] = user

for seller in sellers_set_with_catogories.keys():
    sims = np.zeros(len(u_feature.keys()))
    for i, user in enumerate(u_feature.keys()):
        sims[i] = sim(u_feature[user], sellers_set_with_catogories[seller])
    seller_recommend_user_list[seller] = users[np.argsort(-sims)[0:recommend_user_num]]

with open('results.txt', 'w') as re:
    c_strs = str(seller_recommend_user_list)[1:-1].split(',')
    for c_str in c_strs:
        re.write(c_str + "\n")

