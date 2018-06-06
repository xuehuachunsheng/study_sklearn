#!/usr/local/bin/python3
# -*- coding:utf-8 -*-
# This class is built for logistic regression
__author__ = 'yanxue'

import numpy as np
from numpy import random
from numpy import linalg
from matplotlib import pyplot as plt
import math

class LogisticRegression:
    def __init__(self):
        pass

    def learning(self, data, label, learning_rate=0.1, convergence_rate=1):
        """Learning process of the logistic regression"""
        n, d = data.shape
        self.theta = random.randn(d + 1)
        theta0 = random.randn(d + 1)
        count = 0
        c_error = 0
        for j, x in enumerate(data):
            pLabel = lr.classify(x)
            c_error += pLabel != label[j]
        yield c_error, theta0
        
        while linalg.norm(theta0 - self.theta) > convergence_rate:
            print('theta_{} = {}'.format(count, theta0))
            self.theta = theta0.copy()
            # gradient decent
            theta0 -= learning_rate * sum(map(self.__fun, [(data[i], label[i]) for i in range(n)]))
            # Stochastic gradient descent
            # theta0 -= learning_rate * self.__fun((data[ind % n], label[ind % n]))
            # ind += 1
            # Batch gradient descent
            # theta0 -= learning_rate * sum(map(self.__fun, [(data[i % n], label[i % n]) for i in range(ind, ind + 10)]))
            # ind += 10
            c_error = 0
            for j, x in enumerate(data):
                pLabel = lr.classify(x)
                c_error += pLabel != label[j]
            yield c_error, self.theta
            count += 1

    def __fun(self, trainx):
        return -trainx[1] * np.append(trainx[0], 1) + np.append(trainx[0], 1) * self.s(trainx[0])

    def classify(self, x):
        if self.s(x) < 0.5:
            return 0
        elif self.s(x) >= 0.5:
            return 1

    def s(self, inx):
        """Sigmoid function value"""
        try:
            re = 1 / (1 + math.exp(-np.dot(self.theta, np.append(inx, 1))))
        except OverflowError as e:
            print(e)
            re = 1
        return re

    def __str__(self):
        return "Logistic regression"

trainData = np.loadtxt('./data/classificationData/testSet.txt', delimiter='\t')
trainX = trainData[:, [0, 1]]
trainY = trainData[:, 2]
lr = LogisticRegression()
# ------ Plot the data points ------
x1 = np.arange(-4.0, 3.2, 0.1)
plt.close()
plt.ion()  #interactive mode on
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
# ------ End of plot ------
for c_error, c_theta in lr.learning(trainX, trainY):
    plt.clf()
    # ------ Plot the iterative process ------
    plt.grid()
    plt.scatter(trainX[trainY == 0, 0], trainX[trainY == 0, 1], c='r')  # plot positive instances
    plt.scatter(trainX[trainY == 1, 0], trainX[trainY == 1, 1], c='b')  # plot negative instances
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('Logistic Regression Demo')
    x2 = (-c_theta[0] * x1 - c_theta[2]) / c_theta[1]
    plt.plot(x1, x2)
    plt.text(-4, 12.5, 'errorNum = {}'.format(c_error))
    # ------ End of plot ------
    plt.pause(0.2)
    
input('Program finished. Press any key to continue...')