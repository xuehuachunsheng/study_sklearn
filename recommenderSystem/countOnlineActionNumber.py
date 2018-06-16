#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
# Count the number of online actions of all users
import math
import os
import csv
import numpy as np
import pandas as pd
data_dir = '/Users/wuyanxue/Documents/GitHub/hands-on-ml-with-sklearn-and-TF/data/recommenderData/IJCAI16_shop_user_data'
os.chdir(data_dir)
f = open('taobao.csv')
context = f.readlines()
u_dict = [{}for i in range(2)]
for line in context:
    line = line.replace('\n', '')
    array = line.split(',')
    if int(array[0]) == 0:
        continue
    u_id = (array[0], array[3])
    type1 = int(array[4])
    if (u_id in u_dict[type1]):
        u_dict[type1][u_id] += 1
    else:
        u_dict[type1][u_id] = 1

csvfile = open('feature.csv', 'w')

for i in range(2):
    for key, value in u_dict[i].items():
        key = list(key)
        line = []
        line.append(int(key[0]))
        line.append(int(key[1]))
        line.append(i)
        line.append(value)
        csvfile.write(str(line)[1:-1] + '\n')
csvfile.close()
print('ok')
