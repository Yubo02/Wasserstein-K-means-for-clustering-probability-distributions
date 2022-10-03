# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 07:57:17 2022

@author: yubo0
"""



import torch
torch.set_default_tensor_type(torch.DoubleTensor)
import math
import os
import sys
import numpy as np


from Distribution.Distribution import Distribution
import utils.mnist_utils as mnist
from utils.kmeans_utils import *
from DWKM_utils import *
from utils.sinkhorn_utils import sinkhorn_divergence



# Initialization for D-WKM similar to K-mean+
def initial_plus(chosen_random, data, num_groups, reg):
    centroids = []
    centroids.append(chosen_random)
    for k in range(num_groups - 1):
        distances = np.zeros(len(data))
        for i in range(len(data)):
            distances[i] = sinkhorn_divergence(chosen_random.weights, \
                                               chosen_random.support, \
                                               data[i].weights, data[i].support, eps=reg)[0]

        probs = distances ** 2
        index = np.random.choice(np.arange(0, len(data)), p=probs / sum(probs))
        centroids.append(data[index])
        chosen_random = data[index]
    return centroids




# Substract the data
images = mnist.train_images()
labels = mnist.train_labels()

images1=images[labels==0]
images2=images[labels==5]


images1 = images1[np.random.choice(images1.shape[0], 100)]
images2 = images2[np.random.choice(images2.shape[0], 50)]


num_groups = 2

images = np.concatenate((images1, images2))
num_images0= images.shape[0]



# Transform image to distribution in preparation for the D-WKM
images=images.astype(np.float32)
distrib = []
rescale = 1 / 28
grid=28
reg = 0.001



for i in range(images.shape[0]):
    distrib.append(from_image_to_distribution(images[i], rescale))


###################################################  

# Get the distance matrix

Distance_mat0 = np.full((num_images0, num_images0), 0.0)
for i in range(1,num_images0):
    for j in range(0,i):
        Distance_mat0[i][j] = sinkhorn_divergence(distrib[j].weights, distrib[j].support, \
                                   distrib[i].weights, distrib[i].support, eps=reg)[0]
Distance_mat0 = Distance_mat0 + Distance_mat0.T



# Output the distance matrix to do WSDP in MATLAB           
Distance_mat1=np.matrix(Distance_mat0)
with open('Dis_mat.txt','wb') as f:
    for line in Distance_mat1:
        np.savetxt(f, line, fmt='%.5f')      


###################################################  

# Use initilization similar to K-mean+

ind_rand = np.random.randint(0, num_images0)
first_centroid = from_image_to_distribution(images[ind_rand], rescale)
centroids_distrib = initial_plus(first_centroid,distrib, num_groups,reg)



kmeans_iteration = 0
kmeans_iteration_max = 100
groups_ind0=[]
num_groups = len(centroids_distrib)

groups,groups_ind = partition_into_groups(distrib, centroids_distrib, num_groups,reg,rescale)
print(groups_ind)

while kmeans_iteration < kmeans_iteration_max and groups_ind!=groups_ind0:
    groups_ind0 = groups_ind
    print(groups_ind)
# Update the assignments using D-WKM
    groups_ind =  partition_into_groups_DWKM(distrib,Distance_mat0, groups_ind, num_groups)
    kmeans_iteration = kmeans_iteration + 1
# Get the assignments    
assign_DWKM = groups_ind


        
        




