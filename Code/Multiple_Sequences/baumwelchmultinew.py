#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#multiple input
#new model
##using log trick
#notations are used as in tutorial
#n is length of music
#m is number of different hidden states
#pi is initial distribution
#A is transition matrix
#b is emission matrix
#x is the input matrix of notes
#k is the number of possible notes
#K is the number of input
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
    
#for each single input#
def suboptimize(n, m, k, x, tol):
    #randomly initialize A, b and pi
    #x here is a just one piece
    pi = np.random.rand(m)
    pi = np.log(pi/np.sum(pi))
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
    A=np.random.rand(m,m)
    Adenorm=np.zeros((m,m))
    Anumer=np.zeros((m,m))
    b = np.random.rand(m,m,k)
    bdenorm=np.zeros((m,m,k))
    bnumer=np.zeros((m,m,k))
    A = np.log(A/np.sum(A, axis=1)[:,None])
    b = np.log(b/np.sum(b, axis=(0,1))[None,None,:])
    b0=np.log(b0)
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
            
        criteria = abs(pOld-pNew)
        print("iteration",iterations,"loss",criteria/tol)
        if criteria < tol:
            convergence = 1
        elif iterations > 500:
            convergence = 1
        else:
            convergence = 0
            pOld = pNew
            iterations +=1
    return (pi,b0, bdenorm,bnumer, Adenorm,Anumer)

def optimize(n, m, k, K, x, tol):
    #x here is a list of K pieces, x[K,n] is the last index
    length=[0]*K
    #store the length
    temp=[0]*K
    for i in range(0,K):
        for j in range(0,n):
            if x[i,j]<=0:
                temp[i]=[x[i,j] for j in range(0,j)]
                length[i]=j
                break
            if j==n-1:
                temp[i]=[x[i,j] for j in range(0,n)]
                length[i]=j+1
    x=temp
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
