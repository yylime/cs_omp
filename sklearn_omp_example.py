# -*- coding: utf-8 -*-
"""
Created on Wed May  2 20:05:07 2018

@author: WSPN
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft,ifft
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import OrthogonalMatchingPursuitCV
from sklearn.datasets import make_sparse_coded_signal
import random
import math

'''
x = np.linspace(0,1,20)  
y = y=7*np.sin(2*np.pi*5*x) + 2.8*np.sin(2*np.pi*10*x)
plt.plot(y)
plt.show()
f_y = fft(y)
f_y_r = f_y.real
f_y_i = f_y.imag
abs_f_y = abs(f_y)
plt.plot(abs_f_y)
plt.show()
'''
def gen_fft_matrix(width):
    N = width
    n = np.arange(N).reshape(-1,1)
    n_n = np.dot(n,n.T)
    W = np.exp(-2*np.pi*1j*n_n/N)/np.sqrt(N)
#    print(W)
    return W

'''
n_f = 1024      
#fft_matrix = gen_fft_matrix(n_f)   

#一定的随机化 
fft_matrix = fft(np.eye(n_f))/np.sqrt(n_f)

#给定X频率信号,随机采样得到稀疏信号X
#random.seed(0)
X = np.zeros((n_f,1),dtype = 'complex')
n_nozeros = 2
id_fs = random.sample(range(64),n_nozeros//2)
#id_fs_2 = [n_f-id_f for id_f in id_fs]
X[id_fs,0] = 1
#X[id_fs_2,0] = 1

#X = X[:,0]
#随机采样矩阵
#random.seed(3)
sample_matrix = np.zeros((64,n_f))
id_samples = random.sample(range(n_f),64)
id_samples.sort()
#id_samples.sort()
for i,id_sample in enumerate(id_samples):
    sample_matrix[i,id_sample] = 1

#np.random.seed(3)
#sample_matrix = np.random.randn(65,n_f)


#得到字典，时域信号Y
D = np.dot(sample_matrix,fft_matrix)
#D = sample_matrix
#D = fft_matrix[id_samples,:]

Y = np.dot(D,X)
#Y_real = Y.real
#还原得到X

#取近似
real_id_samples = [round(s/8) for s in id_samples]
real_sample_matrix = np.zeros((64,128))
for i,id_sample in enumerate(real_id_samples):
#    if id_sample==128:
#        id_sample = 127
    real_sample_matrix[i,id_sample] = 1
    
real_fft_matrix  = fft(np.eye(128))/np.sqrt(128)
real_D = np.dot(real_sample_matrix,real_fft_matrix)


omp = OrthogonalMatchingPursuit()
omp.fit(D,Y)
coef = omp.coef_

#近似后的还原
real_omp = OrthogonalMatchingPursuit()
real_omp.fit(real_D,Y)
real_coef = real_omp.coef_

'''
#针对时域信号处理
m = 1
n = 1000
x = np.linspace(0,m,n)
y = 8*np.cos(2*np.pi*20*x) + 16*np.sin(2*np.pi*5*x) + 8*np.sin(2*np.pi*10*x)
e_f = n/m 

random.seed(1)
id_samples = random.sample(range(n),64)
id_samples.sort()

#得到采样的N
id_array = np.array(id_samples).reshape(-1,1)
cf_array = id_array[1:]-id_array[:-1]
td = np.min(cf_array)
N = id_samples[-1]//td + 1
#N = 1000
#N要大于等于数据长度
#N = (int)(e_f*8)

fft_matrix = gen_fft_matrix(N)
#fft_matrix1 = fft(np.eye(N))/np.sqrt(N)
print(fft_matrix.shape)
sample_matrix = np.zeros((64,N))

for i,id_sample in enumerate(id_samples):
    sample_matrix[i,id_sample] = 1

#Y_r = np.dot(sample_matrix,y)
Y_r = y[id_samples]

D = np.dot(sample_matrix,fft_matrix)

#omp = OrthogonalMatchingPursuit()
#omp.fit(D,Y_r)
#coef = omp.coef_

from cs_omp import cs_omp
K = 3
pos_f,theta_ls,theta = cs_omp(Y_r.reshape(-1,1),D,K)


#得到频率值间隔
jg = round(N/e_f)

real_f = []
for i in range(2*K):
    if pos_f[i] > N // 2:
        real_f.append((N-pos_f[i])/jg)
    else:
        real_f.append(pos_f[i]/jg)
  
print(real_f)

#观察还原信号和初始信号
#plt.plot(abs(X)[:100],'r')
plt.plot(abs(theta),'b')
#plt.plot(real_coef[:100],'g')
#plt.show



    
    
    
    
    
    
    
    


