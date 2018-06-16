#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
# Preprocess 'ijcai2016_taobao.csv with removing reduandent users'

import pandas as pd
import os

data_dir = '/Users/wuyanxue/Documents/GitHub/hands-on-ml-with-sklearn-and-TF/data/recommenderData/IJCAI16_shop_user_data'

df1 = pd.read_csv(os.path.join(data_dir, 'ijcai2016_koubei_test'), header=None)
df2 = pd.read_csv(os.path.join(
    data_dir, 'ijcai2016_koubei_train'), header=None)
sets = set(df1[0]) | set(df2[0])
df3 = pd.read_csv(os.path.join(data_dir, 'ijcai2016_taobao.csv'), header=None)
df3 = df3[df3[0].isin(sets)] # Check if a user is in 'ijcai2016_koubei_train.csv or ijcai2016_koubei_test.csv'
df3.to_csv(os.path.join(data_dir, 'taobao.csv'), index=False, header=None)

print('ok')
