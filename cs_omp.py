# -*- coding: utf-8 -*-
"""
Created on Mon May 21 15:04:12 2018

@author: WSPN
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft,ifft
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import OrthogonalMatchingPursuitCV
from sklearn.datasets import make_sparse_coded_signal
import random

def gen_fft_matrix(width):
    N = width
    n = np.arange(N).reshape(-1,1)
    n_n = np.dot(n,n.T)
    W = np.exp(2*np.pi*1j*n_n/N)/N
    return W

def cs_omp(y,D,k):
    '''
    y is the sample signal
    D is the perp matrix D = phi*psi
    K is the Sparsity
    t is the 2*K
    y = D * X ...X is the sparse singla
    '''
    t = 2*k
    #得到输入的长度
    M,N = D.shape
    #待恢复的稀疏信号
    theta = np.zeros((N,1))
    #迭代过程中存储被选择D的列
    Dt = np.zeros((M,t))
    #保存选择D的列的序号
    pos_theta = [] 
    #定义残差
    r_n = y
    
    for i in range (0,t):
        product = np.dot(D.T,r_n)
#        val = np.max(product)
        pos = np.argmax(np.abs(product)) #得到最值及其位置
        Dt[:,i] = D[:,pos]  #保存最大列
        pos_theta.append(pos) #保存最大列的位置
        D[:,pos] = 0 #该列清零
#        print(Dt,Dt.shape)
        temp_A = Dt[:,0:i+1].reshape(M,i+1)
#        print(temp_A.shape)
        temp_I = np.mat((temp_A.T).dot(temp_A)).I
#        print(temp_I.shape)
        theta_ls = np.array(temp_I).dot(temp_A.T).dot(y).reshape(-1,1)
        r_n = y - temp_A.dot(theta_ls)
    print(theta_ls)
    for j,pos in enumerate(pos_theta):
        theta[pos] = theta_ls[j,0]
    
    return pos_theta,theta_ls,theta
    

    

    
    