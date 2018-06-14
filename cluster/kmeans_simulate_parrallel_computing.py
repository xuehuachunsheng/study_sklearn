#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
'''
Simulate the distributed k-means algorithm
'''

from sklearn.datasets import make_blobs
import numpy as np
from matplotlib import pyplot as plt
# Generate simulated data
data = make_blobs(n_samples=4999)
X = data[0]
Y = data[1]
# Global variable
n_samples = 4999
n_dim = 2
k = 3  # number of clusters
centers = np.random.randn(k, n_dim)  # initialize centers
num_nodes = 50  # number of nodes
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


def map(samples):
    '''
    Define each iteration in each nodes

    Input: 
        samples -- the samples in each nodes

    Return: 
        cluster_results -- the cluster indices of all the given samples of the node
        c_centers -- the centers of the node
        n_samples_each_cluster -- the number of samples of each cluster of the node
    '''
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
    return cluster_results, c_centers, n_samples_each_cluster


def reduce(cluster_results, c_centers, n_samples_each_cluster, nodei):
    '''
        Define the reduce progress for all the cluster_results

        Input: 
            cluster_results -- the cluster results of the node
            c_centers -- the centers of the node
            n_samples_each_cluster -- the number of samples of each cluster of the node
            nodei -- the node index
    '''
    cluster_results_all[sample_idxx_range[nodei, 0]:sample_idxx_range[nodei, 1]
                        ], c_centerss[nodei], n_samples_each_cluster_node[nodei] = cluster_results, c_centers, n_samples_each_cluster


def wait():
    '''
        A simulated function that wait other nodes
    '''
    pass


if __name__ == '__main__':
    t_centers = np.random.randn(k, n_dim)
    while abs(np.sum((t_centers - centers)**2)) > 1e-2:
        centers = t_centers
        # Simulate the parallel computing process
        for i in range(num_nodes):
            cluster_results, c_centers, n_samples_each_cluster = map(
                X[sample_idxx_range[i, 0]:sample_idxx_range[i, 1], :])
            reduce(cluster_results, c_centers, n_samples_each_cluster, i)
        wait()
        # End of simulated
        # Recompute centers
        t_centers = np.zeros((k, n_dim))
        for i in range(num_nodes):
            t_n_samples = n_samples_each_cluster_node[i, :].reshape((k, 1))
            t_centers += c_centerss[i] * np.tile(t_n_samples, (1, n_dim))
        t_centers /= np.tile(np.sum(n_samples_each_cluster_node,
                                    axis=0).reshape((k, 1)), (1, n_dim))

    for i in range(k):
        plt.scatter(X[cluster_results_all == i, 0],
                    X[cluster_results_all == i, 1])
    plt.show()
