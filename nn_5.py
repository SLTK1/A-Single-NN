#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 18:35:55 2018

@author: zbw
"""

import numpy as np
from numpy.linalg import cholesky
import matplotlib.pyplot as plt

#setting parameter
sampleNo = 10
dimension = 5
learning_rate = 0.01
sampleNo_test = 30
#Generate random number

mu = 0
sigma = 1
num = dimension + 1
s0 = np.ones(sampleNo)
np.random.seed(0)
s1 = np.random.normal(mu, sigma, sampleNo )
np.random.seed(1)
s2 = np.random.normal(mu, sigma, sampleNo )
np.random.seed(2)
s3 = np.random.normal(mu, sigma, sampleNo )
np.random.seed(3)

s4 = np.random.normal(mu, sigma, sampleNo )
np.random.seed(4)
s5 = np.random.normal(mu, sigma, sampleNo )
np.random.seed(5)
'''
print s1
print s2
print s3
print w
plt.hist(s1, 5, normed=True)
plt.hist(s2, 5, normed=True)
plt.hist(s3, 5, normed=True)
'''

#Learning Algorithm
x = []
x.append(s0)
x.append(s1)
x.append(s2)
x.append(s3)
x.append(s4)
x.append(s5)
#print x
w = np.random.uniform(-1, 1, num)
#print w
d = np.zeros(sampleNo)
for i in range(sampleNo):
    if s1[i] >= 0:
        d[i] = 1
#print d
flag = 1
step = 0
y_ = np.zeros(sampleNo)
while flag > 0 or step > 10000:
    step = step + 1
    flag = 0
    for k in range(sampleNo):
        y = 0
        for j in range (num):
            y = y + w[j] * x[j][k]
        #print y
        if y >= 0:
            y = 1
        else:
            y = 0
        #print y
        y_[k] = y
        delta = learning_rate * (d[k] - y)
        delta_w = np.zeros(num)
        if delta != 0:
            flag = 1
        for i in range(num):
            delta_w[i] = delta * x[i][k]
            w[i] = w[i] + delta_w[i]
    print "step:", step
print "M =", sampleNo
print "learning rate: K0 =", learning_rate
for i in range(num - 1):
    print "w" + str(i), w[i+1]
print "number of iterations:", step
#print d
#print y_



s0_test = np.ones(sampleNo_test)
np.random.seed(6)
s1_test = np.random.normal(mu, sigma, sampleNo_test )
np.random.seed(7)
s2_test = np.random.normal(mu, sigma, sampleNo_test )
np.random.seed(8)
s3_test = np.random.normal(mu, sigma, sampleNo_test )

np.random.seed(9)
s4_test = np.random.normal(mu, sigma, sampleNo_test )
np.random.seed(10)
s5_test = np.random.normal(mu, sigma, sampleNo_test )

x_test = []
x_test.append(s0_test)
x_test.append(s1_test)
x_test.append(s2_test)
x_test.append(s3_test)

x_test.append(s4_test)
x_test.append(s5_test)

d_test = np.zeros(sampleNo_test)
for i in range(sampleNo_test):
    if s1_test[i] >= 0:
        d_test[i] = 1
#print d_test
count = 0
for k in range(sampleNo_test):
    y = 0
    for j in range(num):
        y = y + w[j] * x_test[j][k]
    if y >= 0:
        y = 1
    else:
        y = 0
    if y == d_test[k]:
        count = count + 1

print "accurancy K0 =", count * 1.0 / sampleNo_test