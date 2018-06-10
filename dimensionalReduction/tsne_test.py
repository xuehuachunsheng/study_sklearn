#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn import manifold
from matplotlib import pyplot
import os
import pylab

#data_dir = '/Users/wuyanxue/Documents/GitHub/datasets/getting-started/digit-recognizer/'
data_dir = '/Users/wuyanxue/Documents/References/特征学习/tsne_python'

def dataVisualization_tSNE():
    X = np.loadtxt(os.path.join(data_dir, "mnist2500_X.txt"))
    labels = np.loadtxt(os.path.join(data_dir, "mnist2500_labels.txt"))
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X_tSNE = tsne.fit_transform(X)
    pylab.scatter(X_tSNE[:, 0], X_tSNE[:, 1], 20, labels)
    pylab.show()


if __name__ == '__main__':
    dataVisualization_tSNE()
