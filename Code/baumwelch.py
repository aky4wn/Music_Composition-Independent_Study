#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 10:03:01 2019

@author: mayuheng
"""
##using log trick
#notations are used as in tutorial
#n is number of length of music
#m is number of different hidden states
#pi is initial distribution
#A is transition matrix
#b is emission matrix
#x is the input vector of notes
#k is the number of possible notes
import numpy as np

def logSumExp(a):
    if np.all(np.isinf(a)):
        return np.log(0)
    elif len(a)==0:
        return (np.log(0))
    else:
        b = np.max(a)
    return(b + np.log(np.sum(np.exp(a-b))))

def forward(n, m, pi, A, b, x):
    alpha = np.zeros((n,m))
    for i in range(0,m):
        alpha[0,i] =pi[i] +b[i, x[0]]
    
    for i in range(1, n):
        for j in range(0, m):
            alpha[i,j] = logSumExp(np.asarray(alpha[i-1, :])+np.asarray(A[:,j])+b[j,x[i]])
    return(alpha)
def backward(n, m, pi, A, b, x):
    beta = np.zeros((n,m))
#log1=0 so no initialization
    for t in range(n-2, -1, -1):
        for i in range(0, m):
            beta[t,i] = logSumExp(np.asarray(beta[t+1,: ]) + np.asarray(A[i,:]) + b[:, x[t+1]])
    return(beta)
    

def optimize(n, m, k, x, tol):
#randomly initialize A, b and pi
    pi = np.random.rand(m)
    pi = np.log(pi/np.sum(pi))
    A = np.zeros((m,m))
    b = np.zeros((m,k))
    gamma = np.zeros((n,m))
    xi = np.zeros((n,m,m))
    iterations = 0
    convergence = 0
    count = 0
    pOld = 1E10
    pNew = 0
    
    A=np.random.rand(m,m)
    b=np.random.rand(m,k)
    #slicing
    '''
    '''
    #why???
    A = np.log(A/np.sum(A, axis=1)[:,None])
    b = np.log(b/np.sum(b, axis=1)[:,None])

    #Stop iterations when log(p(x_1:n)) differs by tol between iterations#
    while convergence == 0:
        #Perform forward and backward algorithms# 
        alpha=forward(n,m,pi,A,b,x)
        beta=backward(n,m,pi,A,b,x)
        pNew = logSumExp(alpha[len(x)-1,:])
        
        #Calculate gamma and xi#
        for t in range(0, n):
            for i in range(0,m):
                gamma[t,i] = alpha[t,i] + beta[t,i] - pNew

        for t in range(1, n):
            for i in range(0, m):
                for j in range(0, m):
                    xi[t,i,j] = A[i,j] + b[j, x[t]] + alpha[t-1, i] + beta[t, j] - pNew
                    
        #print([np.sum(np.exp(xi[i,:,:])) for i in range(0,4)])
        #print(np.sum(np.exp(gamma),axis=1))
        #print(np.exp(gamma))
        #print('gamma')
        #Update pi, phi and Tmat#
        #this 1::
        #why xi works gamma not
        
        pi = gamma[0,:] - logSumExp(gamma[0,:])
        for i in range(0, m):
            for j in range(0, m):
                A[i,j] = logSumExp(xi[1::, i, j]) - logSumExp(xi[1::, i,:])
        
        #print(np.exp(A))
        #print('A')
        for i in range(0,m):
            for w in range(0, k):
                j = 0
                count = 0
                for t in range(0,n):
                    if x[t] == w:
                        count = count+1
                indicies = np.zeros(count)
                for t in range(0,n):
                    if x[t] == w:
                        indicies[j] = gamma[t,i]
                        j = j+1
                b[i,w] = logSumExp(indicies) - logSumExp(gamma[:,i])
        
        '''viterbi'''
        #A = np.log(np.exp(A)/np.sum(np.exp(A), axis=1)[:,None])
        #b = np.log(np.exp(b)/np.sum(np.exp(b), axis=1)[:,None])
        #pi = np.log(np.exp(pi)/np.sum(np.exp(pi)))
    
        criteria = abs(pOld-pNew)
        if criteria < tol:
            convergence = 1
        elif iterations > 500:
            convergence = 1
        else:
            convergence = 0
            pOld = pNew
            iterations +=1
            #print(iterations)
    return (iterations, pNew, np.exp(pi), np.exp(b), np.exp(A))