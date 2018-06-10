#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import label_binarize
import pandas as pd
import numpy as np
import os
import time

c_time = time.clock()


# Data path
data_dir = '/Users/wuyanxue/Documents/GitHub/datasets/getting-started/digit-recognizer/'


# Load data
def opencsv():
    print('Load Data...')
    # Open with pandas
    dataTrain = pd.read_csv(os.path.join(data_dir, 'input/train.csv'))
    dataPre = pd.read_csv(os.path.join(data_dir, 'input/test.csv'))
    trainData = dataTrain.values[:, 1:]  
    trainLabel = dataTrain.values[:, 0]
    preData = dataPre.values[:, :]  
    return trainData, trainLabel, preData

# Data
X, y, _ = opencsv()
#X = X[:int(0.2*len(X)), :]
#y = y[:len(X)]
splitRatio = 0.9
X_train = X[:int(splitRatio*len(X)), :]
y_train = y[:int(splitRatio*len(X))]
X_test = X[int(splitRatio*len(X)):, :]
y_test = y[int(splitRatio*len(X)):]


print('Data split')

#param_search_n_estimators = {'n_estimators':range(20, 300, 20)}

#gsearch_gbct = GridSearchCV(GradientBoostingClassifier(), param_grid=param_search_n_estimators, scoring='accuracy', iid=False, cv=5)
gbct = GradientBoostingClassifier(n_estimators=200, subsample=.1)
gbct.fit(X_train, y_train)

print('Fit finished')
#print(gsearch_gbct.grid_scores_, gsearch_gbct.best_params_, gsearch_gbct.best_score_)

y_pre = gbct.predict(X_test)
print(accuracy_score(y_pre, y_test))

print('Time cost: ', time.clock() - c_time)