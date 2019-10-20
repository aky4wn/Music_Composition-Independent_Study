#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 19:58:34 2019

@author: mayuheng
"""
from baumwelchnewmodel import *
import sklearn.metrics
from hmm1order import *
import numpy as np
import scipy.io as scio
hmmcompose('twinkle-twinkle-little-star.csv','generated-twinkle.csv',256,10,1E-5)

n=50#length
m=10#states
k=10#observation
c=10#sample
#randomly set parameter and get out put
A=np.random.rand(m,m)
A=A/np.sum(A, axis=1)[:,None]
b=np.random.rand(k,k)
b=b/np.sum(b, axis=1)[:,None]
pi=np.random.rand(m)
pi=pi/np.sum(pi)
code=range(0,k)

output=[]
states=[]
for i in range(0,c):
    temp1,temp2=generate(n,pi,b,A,code)
    output.append(temp1)
    states.append(temp2)
'''  
ometric1=np.zeros((c,c))
smetric1=np.zeros((c,c))
ometric2=np.zeros((c,c))
smetric2=np.zeros((c,c))
'''
ometric3=np.zeros((2*c,2*c))
smetric3=np.zeros((2*c,2*c))

        
generatedsample=[]
pinew=[]
bnew=[]
Anew=[]
for i in range(0,c):
    iteration,pO,temp1,temp2,temp3=optimize(n, m, k, np.array(output[i],dtype=int), 1E-4)
    pinew.append(temp1)
    bnew.append(temp2)
    Anew.append(temp3)

#what metric to use
generatedoutput=[]
generatedstates=[]
for i in range(0,c):
    temp1,temp2=generate(n,pinew[i],bnew[i],Anew[i],code)
    generatedoutput.append(temp1)
    generatedstates.append(temp2)

#for i in range(0,c):
#    for j in range(0,i):
#        ometric2[i,j]=sklearn.metrics.mutual_info_score(generatedoutput[i],generatedoutput[j])
#        smetric2[i,j]=sklearn.metrics.mutual_info_score(generatedstates[i],generatedstates[j])

dataoutput=output+generatedoutput
datastates=states+generatedstates
for i in range(0,2*c):
    for j in range(0,i):
        ometric3[i,j]=sklearn.metrics.mutual_info_score(dataoutput[i],dataoutput[j])
        smetric3[i,j]=sklearn.metrics.mutual_info_score(datastates[i],datastates[j])
    
data={'A':ometric3}
scio.savemat('/Users/mayuheng/Desktop/A.mat', {'A':data['A']})


from plot import *
Detectionplot()


