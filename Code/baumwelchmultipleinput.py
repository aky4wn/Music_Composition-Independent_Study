#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#multiple input
#first order
##using log trick
#notations are used as in tutorial
#n is length of music
#m is number of different hidden states
#pi is initial distribution
#A is transition matrix
#b is emission matrix
#x is the multiple input matrix of notes
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
    for t in range(n-2, -1, -1):
        for i in range(0, m):
            beta[t,i] = logSumExp(np.asarray(beta[t+1,: ]) + np.asarray(A[i,:]) + b[:, x[t+1]])
    return(beta)
    
#for each single input#
def suboptimize(n, m, k, x, tol):
    #randomly initialize A, b and pi
    #x here is a just one piece
    pi = np.random.rand(m)
    pi = np.log(pi/np.sum(pi))
    gamma = np.zeros((n,m))
    xi = np.zeros((n,m,m))
    iterations = 0
    convergence = 0
    count = 0
    pOld = 1E10
    pNew = 0
    A=np.random.rand(m,m)
    Adenorm=np.zeros((m,m))
    Anumer=np.zeros((m,m))
    b=np.random.rand(m,k)
    bdenorm=np.zeros((m,k))
    bnumer=np.zeros((m,k))
    
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
                    
        
        #Update pi, phi and Tmat #
        pi = gamma[0,:] - logSumExp(gamma[0,:])
        for i in range(0, m):
            for j in range(0, m):
                Anumer[i,j] = logSumExp(xi[1::, i, j]) 
                Adenorm[i,j]=logSumExp(xi[1::, i,:])
                A[i,j]=Adenorm[i,j]-Anumer[i,j]
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
                bnumer[i,w]=logSumExp(indicies)
                bdenorm[i,w]=logSumExp(gamma[:,i])
                b[i,w] = bdenorm[i,w]-bnumer[i,w]
        
        criteria = abs(pOld-pNew)
        if criteria < tol:
            convergence = 1
        elif iterations > 500:
            convergence = 1
        else:
            convergence = 0
            pOld = pNew
            iterations +=1
    return (pi, bdenorm,bnumer, Adenorm,Anumer)

def optimize(n, m, k, K, x, tol):
    #x here is a list of K pieces
    length=[0]*K
    #store the length
    temp=[None]*K
    for i in range(0,K):
        for j in range(0,n):
            if x[i,j]<0:
                temp[i]=x[i,0:j]
                length[i]=j
                break
            if j==n-1:
                temp[i]=x[i]
                length[i]=j+1
    x=temp
    A=np.zeros((2,K,m,m))
    rA=np.zeros((m,m))
    b=np.zeros((2,K,m,k))
    rb=np.zeros((m,k))
    pi=np.zeros((K,m))
    rpi=np.zeros((m))
    for i in range(0,K):
        pi[i],b[0,i],b[1,i],A[0,i],A[1,i]=suboptimize(length[i], m, k, np.ndarray.flatten(np.array(x[i])), tol)
    for i in range(0,m):
        for j in range(0,m):
            rA[i,j]=logSumExp(A[1,:,i,j])-logSumExp(A[0,:,i,j])
    for i in range(0,m):
        for j in range(0,k):   
            rb[i,j]=logSumExp(b[1,:,i,j])-logSumExp(b[0,:,i,j])
    for i in range(0,m):
        rpi[i]=logSumExp(pi[:,i])-np.log(K)
    return (np.exp(rA),np.exp(rb),np.exp(rpi))
    
