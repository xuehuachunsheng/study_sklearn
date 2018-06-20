#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
'''
kmeans并行计算算法，利用了进程池回调机制。
算法思想：假设有n个样本需要聚类，现有m个pc机(节点)，那么按照等分(或近似等分)的手段将这n个样本分成m份，
每一份分配给子进程，子进程会在迭代的时候计算部分样本的中心，并返回每个簇的样本个数，这是为了方便
进行全局样本中心的更新。在主进程拿到所有节点的中心和对应的样本个数，就可以在在主进程上做全局中心的更新。
全局变量
    样本数量 -- n_samples
    样本维度 -- n_dim
    全局中心 -- centers
    子进程个数 -- num_nodes
    子进程将要计算的样本下标范围（或者在文件中的行号范围） -- sample_idxx_range
    聚类结果 -- cluster_results_all
    所有子进程所计算的中心 -- c_centerss
    所有子进程所计算的每个簇的样本个数 -- n_samples_each_cluster_node

'''

from sklearn.datasets import make_blobs
import numpy as np
from matplotlib import pyplot as plt
import multiprocessing as ms
import os
import time

np.seterr(divide='ignore',invalid='ignore')

# Generate simulated data
data = make_blobs(n_samples=50000)
X = data[0]
Y = data[1]
# Global variable
n_samples = 50000
n_dim = 2
k = 3  # number of clusters
centers = np.random.randn(k, n_dim)  # initialize centers
num_nodes = 20  # number of nodes
# Record sample indices of each node
n_samples_each_nodes = int(n_samples / num_nodes)
sample_idxx_range = np.zeros((num_nodes, 2), dtype=int)
c_index = 0
for i in range(num_nodes - 1):
    sample_idxx_range[i, 0] = c_index
    sample_idxx_range[i, 1] = c_index + n_samples_each_nodes
    c_index += n_samples_each_nodes
sample_idxx_range[num_nodes - 1, 0] = c_index
sample_idxx_range[num_nodes - 1, 1] = n_samples

# cluster results of all samples in the current iteration
cluster_results_all = np.zeros(n_samples, dtype=int)
c_centerss = np.zeros((num_nodes, k, n_dim))  # current centers of each node
# number samples of each cluster in each node
n_samples_each_cluster_node = np.zeros((num_nodes, k), dtype=int)
# End of global variable

def map(a):
    '''
    Compute the centers of the part of samples

    Input: 
        samples -- the samples in each nodes

    Return: 
        cluster_results -- the cluster indices of all the given samples of the node
        c_centers -- the centers of the node
        n_samples_each_cluster -- the number of samples of each cluster of the node
    '''
    samples = a[0]
    nodei = a[1]
    cluster_results = np.zeros(len(samples), dtype=int)
    n_samples_each_cluster = np.zeros(k, dtype=int)
    c_centers = np.zeros((k, n_dim))
    for i, sample in enumerate(samples):
        t_dists = np.sum((np.tile(sample, (k, 1)) - centers) **
                         2, axis=1)  # Square of euclidean distance
        cluster_results[i] = np.argmin(t_dists)
        n_samples_each_cluster[cluster_results[i]] += 1
        c_centers[cluster_results[i]] += sample
    for i, m in enumerate(n_samples_each_cluster):
        if m != 0:
            c_centers[i] = c_centers[i] / m
    return cluster_results, c_centers, n_samples_each_cluster, nodei

def reduce(result):
    # To change global variable.
    #global cluster_results_all, c_centerss, n_samples_each_cluster_node 
    cluster_results, c_centers, n_samples_each_cluster, nodei = result
    # There is no neccesary to add lock
    cluster_results_all[sample_idxx_range[nodei, 0]:sample_idxx_range[nodei, 1]
                            ], c_centerss[nodei], n_samples_each_cluster_node[nodei] = cluster_results, c_centers, n_samples_each_cluster

if __name__ == '__main__':
    avg_time = 0
    
    c_time = time.clock()
    t_centers = np.random.randn(k, n_dim)
    while abs(np.sum((t_centers - centers)**2)) > 1e-2:
        centers = t_centers
        # Parallel computing process
        processS = ms.pool.Pool(num_nodes)
        
        for i in range(num_nodes):
            a = (X[sample_idxx_range[i, 0]:sample_idxx_range[i, 1], :], i)
            processS.apply_async(map, args=(a, ), callback=reduce)
        processS.close() # Wait all the sub processes 
        processS.join() # Main process blocks until all sub processes
        # End of Parallel computing process
        
        # Recompute centers
        t_centers = np.zeros((k, n_dim))
        for i in range(num_nodes):
            t_n_samples = n_samples_each_cluster_node[i, :].reshape((k, 1))
            t_centers += c_centerss[i] * np.tile(t_n_samples, (1, n_dim))
        t_centers /= np.tile(np.sum(n_samples_each_cluster_node,
                                    axis=0).reshape((k, 1)), (1, n_dim))

    print('time costs: ', time.clock() - c_time)

    for i in range(k):
        plt.scatter(X[cluster_results_all == i, 0],
                    X[cluster_results_all == i, 1])
    plt.show()
