#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#the new model
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

def forward(n, m, pi, A, b,b0, x):
    alpha = np.zeros((n,m))
    for i in range(0,m):
        alpha[0,i] =pi[i] +b0[ i, x[0]]
    
    for i in range(1, n):
        for j in range(0, m):
            alpha[i,j] = logSumExp(np.asarray(alpha[i-1, :])+np.asarray(A[:,j])+b[:,j,x[i]])
    return(alpha)
def backward(n, m, pi, A, b, x):
    beta = np.zeros((n,m))
    
    for t in range(n-2, -1, -1):
        for i in range(0, m):
            beta[t,i] = logSumExp(np.asarray(beta[t+1,: ]) + np.asarray(A[i,:]) + b[i,:, x[t+1]])
    return(beta)
    

def optimize(n, m, k, x, tol):
    #randomly initialize A, b and pi
    pi = np.random.rand(m)
    pi = np.log(pi/np.sum(pi))
    A = np.random.rand(m,m)
    b = np.random.rand(m,m,k)
    b0=np.zeros((m,k))
    for i in range(0,m):
        b0[i,x[0]]=1
    gamma = np.zeros((n,m))
    xi = np.zeros((n,m,m))
    iterations = 0
    convergence = 0
    count = 0
    pOld = 1E10
    pNew = 0
    
    
    A = np.log(A/np.sum(A, axis=1)[:,None])
    b = np.log(b/np.sum(b, axis=(0,1))[None,None,:])
    b0= np.log(b0)
    #Stop iterations when log(p(x_1:n)) differs by tol between iterations#
    while convergence == 0:
        #Perform forward and backward algorithms# 
        alpha=forward(n,m,pi,A,b,b0,x)
        beta=backward(n,m,pi,A,b,x)
        pNew = logSumExp(alpha[len(x)-1,:])
        
        #Calculate gamma and xi#
        for t in range(0, n):
            for i in range(0,m):
                gamma[t,i] = alpha[t,i] + beta[t,i] - pNew

        for t in range(1, n):
            for i in range(0, m):
                for j in range(0, m):
                    xi[t,i,j] = A[i,j] + b[i,j, x[t]] + alpha[t-1, i] + beta[t, j] - pNew
        
        pi = gamma[0,:] - logSumExp(gamma[0,:])
        for i in range(0, m):
            for j in range(0, m):
                A[i,j] = logSumExp(xi[1::, i, j]) - logSumExp(xi[1::, i,:])

        for i in range(0,m):
            for j in range(0,m):
                for w in range(0, k):
                    h = 0
                    count = 0
                    for t in range(1,n):
                        if x[t] == w:
                            count = count+1
                    indicies = np.zeros(count)
                    for t in range(1,n):
                        if x[t] == w:
                            indicies[h] = xi[t,i,j]
                            h = h+1
                    b[i,j,w] = logSumExp(indicies) - logSumExp(xi[1::,i,j])
        

        criteria = abs(pOld-pNew)
        if criteria < tol:
            convergence = 1
        elif iterations > 500:
            convergence = 1
        else:
            convergence = 0
            pOld = pNew
            iterations +=1
    return (np.exp(pi), np.exp(b0) ,np.exp(b), np.exp(A))