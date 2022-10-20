# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 08:16:00 2022

@author: yubo0
"""


import torch
import numpy as np
from utils.sinkhorn_utils import sinkhorn_divergence



# Define bregman-Wasserstein Barycenter

def bregmanWassersteinBarycenter(D, GM, _lambda, w, _iter):
    N = D.shape[0]     # number of pixels
    K = D.shape[1]    # number of histograms
    xi = np.exp(-_lambda*GM)
    u = np.asmatrix(np.zeros((N, K)))+1
    v = np.asmatrix(np.zeros((N, K)))+1
    c = np.zeros((1, N))
    flag = 0
    for i in range(_iter):
        u_xi_v = np.asmatrix(np.zeros((N, K)))
        c = np.zeros((N, 1))
        for k in range(K):
            u_xi_v[:,k] = np.log( np.multiply(u[:,k], (xi * v[:,k]))  )
            c = c + w[0,k] * u_xi_v[:,k]
        c = np.exp(c)
        for k in range(K):
            u[:,k] = c/(xi * v[:,k])
            v[:,k] = D[:,k]/(xi.T * u[:,k])
    # reshape the barycenter to 2D
    n = round(np.sqrt(N))
    C = np.asmatrix(np.zeros((n, n)))
    flag = 0
    for j in range(n):
        for i in range(n):
            C[i,j] = c[flag]
            flag = flag + 1
    return C



# In preparation for bregman-Wasserstein Barycenter

def computeDistanceMatrixGrid(n):
    nn = np.square(n)
    M = np.asmatrix(np.zeros((nn, nn)))
    x = np.asmatrix(np.zeros((nn, 2)))
    lin_grid = np.linspace(0, 1, n)    # linear grid
    flag = 0
    for i in range(n):
        for j in range(n):
            x[flag,:] = [lin_grid[i], lin_grid[j]]
            flag = flag + 1
    y = x
    for i in range(nn):
        for j in range(nn):
            M[i,j] = np.sum( np.square(x[i,:]-y[j,:]) )   # squared Euclidean distance
    return M




#################################################################################
# To run D-WKM 
# Assign the distribution to the cluster that minimizes the averaged squared W2 distances

def partition_into_groups_DWKM(data,Di, groups_ind_0, num_groups):
    groups_ind = [[] for i in range(num_groups)]
    for i in range(len(data)):
        min_dist = 10000
        for k in range(len(groups_ind_0)):
            dist = sinkhorn_divergence_cluster(groups_ind_0[k],i,Di)

            if dist < min_dist:
                tmp_c = k
                min_dist = dist
        groups_ind[tmp_c].append(i)
    return groups_ind

# Calculate the averaged squared W2 distances for distribution ii to the cluster indexed by groups_ind_0_0.
def sinkhorn_divergence_cluster(groups_ind_0_0,ii,Di):
    dist1=0
    for i in range(len(groups_ind_0_0)):
        dist2= Di[groups_ind_0_0[i]][ii]
        dist1=dist1+dist2
    dist3=dist1/len(groups_ind_0_0)
    return dist3




# Assign the distribution to the nearest cluster based on the cetroids

def partition_into_groups_withind(data, centroids, num_groups, reg, rescale):
    groups = [[] for i in range(num_groups)]
    groups_ind = [[] for i in range(num_groups)]
    for i in range(len(data)):
        min_dist = 100
        for k in range(len(centroids)):
            dist = sinkhorn_divergence(centroids[k].weights, centroids[k].support, \
                                       data[i].weights, data[i].support, eps=reg)[0]

            if dist < min_dist:
                tmp_c = k
                min_dist = dist

        groups[tmp_c].append(data[i])
        groups_ind[tmp_c].append(i)
    return groups,groups_ind

