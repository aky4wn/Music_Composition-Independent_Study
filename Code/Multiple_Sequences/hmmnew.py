#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import baumwelchmultinew
import baumwelchnewmodel
import numpy as np
import csv
'''
A and pi are transition matrix and innitial distribution
b and b0 are emission distribution and innitial emission distribution
generally n represents length of a piece, m represents number of hidden states
k represents number of unique notes, multiple represents number of input pieces
notes is input multiple pieces, note that one piece is one of its colume 
cut is cutting last pieces, used for debugging
''' 
    
def hmmnewcompose(notes,m,tol,multiple,cut=0):
    print("GOGOGOGO!")
    if cut>0:
        multiple=multiple-cut
    possibleNotes=[]
    # decide single or multiple input
    if multiple>1:
        possibleNotes=np.unique(np.ravel(np.concatenate([np.unique(notes.T[i]) for i in range(0,multiple)])))
        k=len(possibleNotes)-1
        xNotes=notes.T
        xNotes=xNotes[0:multiple]
        xNotes=xNotes-1
        n=xNotes.shape[1]-1
        A,b0,b,pi=baumwelchmultinew.optimize(n,m,k,multiple,xNotes,tol)
    else:
        possibleNotes=np.unique(notes)
        k=len(possibleNotes)
        n=len(xNotes)
        pi,b0,b,A=baumwelchnewmodel.optimize(n,m,k,notes,tol)
    return(A,b0,b,pi)
    