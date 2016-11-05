# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 00:51:16 2016

@author: alvin
"""

from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from sklearn.mixture import GMM
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as stats
from scipy.stats import gamma, norm, uniform
from functools import partial
from scipy.special import gamma
def lambda_to_std(_lambda):
    return 1/(np.sqrt(_lambda))
    
def count_unique(val, vec):
    counter = 0
    for _val in vec:
        if val == _val:
            counter += 1
    return counter 


def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

def runningMeanFast(x, N):
    return np.convolve(x, np.ones((N,))/N)[(N-1):]
    
np.random.seed(1234)

gmm = GMM(3, n_iter=1)
gmm.means_ = np.array([[-12], [0], [12]])
gmm.covars_ = np.array([[1], [1], [1]]) ** 2
gmm.weights_ = np.array([0.334, 0.333, 0.333])

Y = gmm.sample(50)
Y = Y.flatten()
#plt.hist(Y, bins = 50)

epsilon = 0.5
s = 0.1
s_inv = 1/s
s_sqrt = np.sqrt(s_inv)
#s_sqrt = s_inv
a = 0.1
b = 0.1 
c = 1
nu = np.random.beta(c+1, len(Y))

#cycle 1
#initalize first weights
w = []
w.append(np.random.beta(1, c))

v = []
v.append(w[0])
#initalize first mu
mus = []
mus.append(np.random.normal(0, s_sqrt))

#initalize first lambda
lambdas = []
lambdas.append(np.random.gamma(epsilon,epsilon))
stds = []
stds.append(lambda_to_std(lambdas[0]))
z = np.zeros(len(Y)) #cluster assignments
u = np.zeros(len(Y))
active_ks = []
total_ks = []
total_cs = []

for cur_iter in range(10000):
    #print "Starting iter %i"%(cur_iter)
    u = np.zeros(len(Y))
    #grab u 
    for idx, val in enumerate(Y):
        cluster_label = int(z[idx])
        u[idx] = np.random.uniform(0, w[cluster_label])
    
    #min u 
    u_star = np.min(u)
    
    #find k_star and initalize components until we have enough
    beta_star = 1 - np.sum(w)
    if cur_iter > 1 and cur_iter % 50 == 0:
        print "On iteration %i\tBeta_star is %0.06f\tU-star is %0.06f\tActive ks %i\tAlpha is %0.06f\n"%(cur_iter, beta_star, u_star, active_ks[-1], total_cs[-1])
        print w
    while (beta_star >= u_star):
        _vk = np.random.beta(1, c)
        beta_k = _vk * beta_star
        
        #append beta_k to the weights
        w.append(beta_k)
            
        #append _vk to v
        v.append(_vk)
        
        #append new mu to mus 
        mus.append(np.random.normal(0, s_sqrt))
        
        #append new lambda to lambda
        new_lambd = np.random.gamma(epsilon, 1/epsilon)
        lambdas.append(new_lambd)

        
        #finally update beta_star
        beta_star = beta_star*(1 -_vk)
    
    assert len(w) == len(mus)
    assert len(w) == len(lambdas)
    
    #now we get the cluster assignments z for each observation
    for y_idx , y_val in enumerate(Y):
        z_probs = np.zeros(len(w))
        for cluster_idx, mu in enumerate(mus):
            if w[cluster_idx] > u[y_idx]:
                z_probs[cluster_idx] = norm.pdf(y_val, loc = mu, scale = lambda_to_std(lambdas[cluster_idx]))
            else:
                z_probs[cluster_idx] = 0
        #sample cluster assignments
        norm_z_probs = z_probs / np.sum(z_probs)
        z_sampler = stats.rv_discrete(name = "z_sampler", values = (np.arange(len(z_probs)), norm_z_probs))
        z_assignment = z_sampler.rvs()
        z[y_idx] = z_assignment
        #print "%i\t%0.03f\t%0.03f\t%0.03f\t%i"%(y_idx, y_val, mu_)
    
    z_ints = [int(x) for x in z] #make z ints for easier use in future 
    
    #perform necessary updates for means and lambdas for each cluster 
    new_mus = []
    new_lambdas = []
    new_z = np.zeros(len(Y))
    active_cluster = 0
    count_dict = {}
    
    for cluster_idx in range(len(w)):
        #update means
        cur_lambda = lambdas[cluster_idx]    
        eta = 0
        m = 0
        y_k = []
        for y_idx, z_val in enumerate(z_ints):
            if z_val == cluster_idx:
                m += 1
                eta += Y[y_idx]
                y_k.append(Y[y_idx])
                new_z[y_idx] = active_cluster
        
        if m == 0:
            continue
        else:
            count_dict[active_cluster] = m
            active_cluster += 1        
            _mean = (eta * cur_lambda)/(m * cur_lambda + s)
            _variance = 1 / (m * cur_lambda + s)
            new_mu = np.random.normal(_mean, np.sqrt(_variance))
            #mus[cluster_idx] = new_mu
            new_mus.append(new_mu)
            #update lambda
            d = 0
            for val in y_k:
                d += (val - new_mu) ** 2
            
            _alpha = epsilon + m/2
            _beta = epsilon + d/2
            new_lambda = np.random.gamma(_alpha, 1/_beta)
            #lambdas[cluster_idx] = new_lambda
            #print "added at %i"%(cluster_idx)
            new_lambdas.append(new_lambda)
            stds.append(lambda_to_std(new_lambda))
            
    #update the stick breaking lengths v_k 
    k = active_cluster 
    vks = []
    #print count_dict
    for cluster_idx in range(k):
        n_k = count_dict[cluster_idx]
        n_j = 0
        #print range(cluster_idx + 1, k )
        for i in range(cluster_idx+1, k):
            n_j += count_dict[i]
        vks.append(np.random.beta(1 + n_k, c + n_j))
        
    #update weights w_k
    new_w = []
    old_vks = []
    for cluster_idx in range(k):
        if cluster_idx == 0:
            new_w.append(vks[cluster_idx])    
            old_vks.append(1-vks[cluster_idx])
        else:
            #print cluster_idx
            vj = 1 
            for i in range(cluster_idx):
                #print i
                #print old_vks
                vj *= old_vks[i]
            new_w.append(vks[cluster_idx] * vj)
            old_vks.append(1-vks[cluster_idx])   
    
    #lastly, update c (alpha) and nu (Escobar & West 1995, section 6)
    #update c
    new_nu = np.random.beta(c + 1, len(Y))    
    
    pi_n_ratio = (a + k - 1) / (len(Y) * (b - np.log10(new_nu)))
    pi_1 = pi_n_ratio / (1 + pi_n_ratio)
    pi_2 = 1 - pi_1
    #new_c = pi_1 * np.random.gamma(a+k, b - np.log10(new_nu)) + pi_2 * np.random.gamma(a+k-1, b-np.log10(new_nu))
    new_c = pi_1 * np.random.gamma(a+k, b - np.log10(new_nu)) + pi_2 * np.random.gamma(a+k-1, b-np.log10(new_nu))
    #new_c = np.random.gamma(a+k, b - np.log10(new_nu))
    #new_c = ((c**k )* gamma(c) * np.random.gamma(0.1,10)) / (gamma(c + len(Y)))      
    
    #store results
    active_ks.append(k)
    total_ks.append(len(w))
    total_cs.append(new_c)
    
    #end of gibbs cycle, update all the variables for next round
    mus = new_mus
    lambdas = new_lambdas
    z = new_z 
    c = new_c
    nu = new_nu
    v = vks 
    w = new_w

plt.clf()
plt.scatter(z , Y)
plt.title('Representative Clustering')
plt.xlabel('Cluster Index')
plt.ylabel('Y value')
plt.savefig('rep_clustv.png',dpi=450)


plt.clf()
#plt.scatter(np.arange(len(active_ks)), active_ks)

plt.plot(np.arange(len(active_ks)), movingaverage(total_ks, 500), 'b-', label='Moving average total k')
plt.plot(np.arange(len(active_ks)), movingaverage(active_ks, 500), 'r-', label='Moving average realized k')
plt.xlabel('Iterations')
plt.ylabel('Clusters')
plt.xlim((0,5000))
plt.legend(prop={'size':8})
plt.title('Moving average of clusters vs. Gibbs Iterations')
plt.savefig('var_c2.png', dpi=500)


plt.plot(np.arange(len(active_ks)), total_cs)

plt.clf()
plt.hist(Y, bins=35)
plt.xlabel('Y value')
plt.ylabel('Count')
plt.title('Distribution on Y')
plt.savefig('ydist.png', dpi=500)