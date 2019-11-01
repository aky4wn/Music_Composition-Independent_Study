#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
multiple inputs for new model
using log trick
A and pi are transition matrix and innitial distribution
b and b0 are emission distribution and innitial emission distribution
generally n represents length of a piece
m represents number of hidden states
k represents number of unique notes
K represents number of input pieces
x is the input matrix of notes
'''
import numpy as np

# function for log addition#
def logSumExp(a):
    if np.all(np.isinf(a)):
        return np.log(0)
    elif len(a)==0:
        return (np.log(0))
    else:
        b = np.max(a)
        return(b + np.log(np.sum(np.exp(a-b))))

# forward and backward algorithm using given parameters#
def forward(n, m, pi, A, b,b0, x):
    alpha = np.zeros((n,m))
    for i in range(0,m):
        alpha[0,i] =pi[i] +b0[i, x[0]]
    
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

# tol is the tolerance of iteration error#
# perform optimize for single pieces#
# returning pi, b0#
# returning matrix of numerator and denominator of b, as to take average separately and devide#
# returning matrix of numerator and denominator of b, as to take average separately and devide#
def suboptimize(n, m, k, x, tol):
    #innitializing 
    pi = np.random.rand(m)
    #setting all parameters in b0 to possitive to avoid#
    #limited renewal for innitial emission distribution#
    b0=np.full((m,k),1E-2)
    for i in range(0,m):
        b0[i,x[0]]=1
        b0[i] = b0[i]/np.sum(b0[i])
    A=np.random.rand(m,m)   
    b = np.random.rand(m,m,k)
    # trasition expectations#
    gamma = np.zeros((n,m))
    xi = np.zeros((n,m,m))
    # other parameters used for iteration#
    # iterations counts iteration times#
    # convergence decide whether reached convergence condition#
    # count is used for computing b#
    # pOld and pNew represent p(x_1:n)#
    iterations = 0
    convergence = 0
    count = 0
    pOld = 1E10
    pNew = 0
    # to store denominator and numerator of A and b#
    Adenorm=np.zeros((m,m))
    Anumer=np.zeros((m,m))
    bdenorm=np.zeros((m,m,k))
    bnumer=np.zeros((m,m,k))
    A = np.log(A/np.sum(A, axis=1)[:,None])
    b = np.log(b/np.sum(b, axis=(0,1))[None,None,:])
    b0=np.log(b0)
    pi = np.log(pi/np.sum(pi))
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
                    
        
        #Update pi, phi and Tmat #
        pi = gamma[0,:] - logSumExp(gamma[0,:])
        for i in range(0, m):
            for j in range(0, m):
                Anumer[i,j] = logSumExp(xi[1::, i, j]) 
                Adenorm[i,j]=logSumExp(xi[1::, i,:])
                A[i,j]=Anumer[i,j]-Adenorm[i,j]
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
                    bnumer[i,j,w]=logSumExp(indicies)
                    bdenorm[i,j,w]=logSumExp(xi[1::,i,j])
                    b[i,j,w] = bnumer[i,j,w]-bdenorm[i,j,w]
        # decide convergence# 
        criteria = abs(pOld-pNew)
        print("iteration",iterations,"difference",criteria/tol,"pnew",pNew)
        if criteria < tol:
            convergence = 1
        elif iterations > 500:
            convergence = 1
        else:
            convergence = 0
            pOld = pNew
            iterations +=1
    return (pi,b0, bdenorm,bnumer, Adenorm,Anumer)

# tol is the tolerance of iteration error#
# perform optimize for multiple sequence using suboptimize#
# input is matrix with one piece in each row and keys to be nonnegative integers#
# returning optimized parameters#
def optimize(n, m, k, K, x, tol):
    # get single sequence to proper length#
    length=[0]*K
    temp=[0]*K
    for i in range(0,K):
        for j in range(0,n):
            if x[i,j]<0:
                temp[i]=[x[i,l] for l in range(0,j)]
                length[i]=j
                break
            if j==n-1:
                temp[i]=[x[i,l] for l in range(0,n)]
                length[i]=j+1
    x=temp
    # A,b,pi and b0 store the result from suboptimize#
    # rA, rb, rb0 and rpi are final result#
    A=np.zeros((2,K,m,m))
    rA=np.zeros((m,m))
    b=np.zeros((2,K,m,m,k))
    rb=np.zeros((m,m,k))
    pi=np.zeros((K,m))
    rpi=np.zeros((m))
    b0=np.zeros((K,m,k))
    rb0=np.zeros((m,k))
    for i in range(0,K):
        print("This is",i+1,"th composing")
        pi[i],b0[i],b[0,i],b[1,i],A[0,i],A[1,i]=suboptimize(length[i], m, k, x[i], tol)
    # take average and divide#
    for i in range(0,m):
        for j in range(0,m):
            rA[i,j]=logSumExp(A[1,:,i,j])-logSumExp(A[0,:,i,j])
    for i in range(0,m):
        for j in range(0,m):
            for w in range(0,k):
                rb[i,j,w]=logSumExp(b[1,:,i,j,w])-logSumExp(b[0,:,i,j,w])
    for i in range(0,m):
        rpi[i]=logSumExp(pi[:,i])-np.log(K)
    for i in range(0,m):
        for j in range(0,k):
            rb0[i,j]=logSumExp(b0[:,i,j])-np.log(K)
    return (np.exp(rA),np.exp(rb0),np.exp(rb),np.exp(rpi))
