#coding:utf-8
import numpy as np
import random
import matplotlib.pyplot as plt
import heapq
#from numpy import linalg
#K-Means聚类函数
def K_means(data_array,k):
    mu_array = np.zeros((k,data_array.shape[1]))
    mu_index = random.sample(range(0, data_array.shape[0]), k)
    for index,value in enumerate(mu_index):
        #mu_array[index] = np.random.rand(k)*(max(data_array[:,i]-min(data_array[:,i])))+min(data_array[:,i])
        mu_array[index] = data_array[value]
    class_index = np.zeros((data_array.shape[0]))
    for train_iters in range(0,100):
        #print(train_iters)
        class_index_tmp = class_index.copy()
        #print(class_index_tmp)
        class_sum = np.zeros((k,data_array.shape[1]+1))
        for j in range(0,data_array.shape[0]):
            distance = []
            for t in range(0,k):
                distance.append(np.linalg.norm(np.array(data_array[j,:])-mu_array[t]))
            class_index[j] = np.argmin(np.array(distance))
            class_sum[class_index[j],0:data_array.shape[1]] = class_sum[class_index[j],0:data_array.shape[1]] + data_array[j,:]
            class_sum[class_index[j], data_array.shape[1]] = class_sum[class_index[j], data_array.shape[1]] + 1
        #print(class_index_tmp)
        if((class_index == class_index_tmp).all()):
            #print(class_index_tmp)
            break
        else:
            for t in range(0, k):
                mu_array[t] = class_sum[t,0:data_array.shape[1]]/class_sum[t, data_array.shape[1]]
    return class_index,mu_array
#谱聚类函数
def Spectral_clustering(data_array,sigma=2,w_k=5,l_k=3,m_k=2):
    W = np.zeros((data_array.shape[0],data_array.shape[0]))
    D = np.zeros((data_array.shape[0], data_array.shape[0]))
    D_half = np.zeros((data_array.shape[0], data_array.shape[0]))

    for i in range(0,data_array.shape[0]):
        for j in range(i+1,data_array.shape[0]):
            W[i,j] = np.e**(-(np.linalg.norm(data_array[i]-data_array[j])**2)/(2*sigma**2))
            W[j,i] = W[i,j]
            #print(W[j,i])
    #print(W[0])
    #print(W[:,0])
    for i in range(0,data_array.shape[0]):
        lager_index = heapq.nsmallest(data_array.shape[0]-w_k-1, range(len(W[i])), W[i].take)
        #print(len(lager_index))
        W[i][lager_index] = 0
    #print(W[0])
    W = (W + W.T)/2
    #print(W[0])
    for i in range(0, data_array.shape[0]):
        D[i, i] = np.sum(W[i])
        D_half[i,i] = np.sum(W[i])**(-0.5)
    #print(D_half[1])

    L = D - W
    #print(L[0])
    L_sym = np.dot(np.dot(D_half,L),D_half)
    #print(np.linalg.det(L_sym))
    v,U = np.linalg.eig(L_sym)
    #print(v[3])
    #print(v.shape)
    v = abs(v)
    #v = np.real(v)
    small_index = heapq.nsmallest(l_k,range(len(v)), v.take)
    #print(v[small_index])
    T = np.zeros((data_array.shape[0], l_k))
    for index,value in enumerate(small_index):
        #print(U[value])
        T[:,index] = abs((U[value]))
    for i in range(0,data_array.shape[0]):
        T_i_tmp = np.linalg.norm(T[i])
        if (T_i_tmp == 0):
            continue
        T[i] = T[i]/T_i_tmp
    #print(T)
    class_index, mu_array = K_means(T,m_k)

    return class_index,mu_array







