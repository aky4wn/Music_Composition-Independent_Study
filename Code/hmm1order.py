#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 22:35:57 2019

@author: mayuheng
"""
from baumwelch import *
import numpy as np
import csv
import pandas as pd
from scipy.interpolate import UnivariateSpline

#is this for memory efficiency?
''''''
def encode(x, code):
    output = np.array([int(np.where(code == x[i])[0]) for i in range(0,len(x))])
    return output
def decode(x,code):
    output = np.zeros(len(x)) 
    for i in range(0, len(x)): 
        output[i] = code[x[i]]
    return output

def generate(n,pi,b,A,code): 
    m = A.shape[0]
    k = b.shape[1]
    ostates=range(0,k)
    xstates=range(0,m)
    o=np.zeros(n,dtype=int)
    x=np.zeros(n,dtype=int)
    x[0]=np.random.choice(xstates,p=pi)
    for j in range(1,n):
        x[j]=np.random.choice(xstates,p=A[x[j-1],:])
    for j in range(0,n):
        o[j]=np.random.choice(ostates,p=b[x[j],:])
    output=decode(o,code)
    return (output,x)
    
#cant understand, sorry for just copying

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin() 
    return array[idx]

class pre_process(object):
    def __init__(self, input_filename, min_note):
            self.input_filename = input_filename
            self.min_note = min_note
    def read_process(self):
        with open(self.input_filename,encoding = "ISO-8859-1") as fd:
            reader=csv.reader(fd)
            rows= [row for idx, row in enumerate(reader)]
        song = pd.DataFrame(rows)
        r,c = np.where(song == ' Header')
        quarter_note = song.iloc[r,5].values.astype(int)[0] 
        r, c = np.where(song == ' Time_signature')
        num = song.iloc[r, 3].values.astype(int)[0]
        denom = song.iloc[r, 4].values.astype(int)[0]**2 
        try:
                    r, c = np.where(song == ' Key_signature')
                    key = song.iloc[r,3].values.astype(int)[0] 
        except:
                    key = None
                    
        song_model = song.loc[song.iloc[:,0] == np.max(song.iloc[:,0])]
        song_model = song_model[song_model.iloc[:, 2].isin([' Note_on_c', 'Note_off_c'])]
        time = np.array(song_model.iloc[:,1]).astype(int)
        notes = np.array(song_model.iloc[:,4]).astype(int)
        velocity = np.array(song_model.iloc[:,5]).astype(int)
        measures = np.round(np.max(time)/quarter_note)/num
        min_note = quarter_note
        actual = np.arange(0, min_note*measures*num, min_note).astype(int)
        time = np.array([find_nearest(actual, time[i]) for i in range(len(time))]).astype(int)
        return(quarter_note, num, denom, key, measures, time, notes, velocity,song, song_model.index)
        
#interpolation
def find_vel(newNotes, velocity):
    newVelocities=np.zeros(len(newNotes))
    y=velocity[np.nonzero(velocity)]
    indecies=[]
    for i in np.unique(newNotes):
        indecies.append(np.where(newNotes==i)[0][::2])
    unlist = [item for sublist in indecies for item in sublist]
    unlist.sort()
    X = np.array(range(0,len(y)))
    s = UnivariateSpline(X, y, s=300) #750
    xs = np.linspace(0, len(y), len(unlist), endpoint = True)
    ys = s(xs)
    newVelocities[np.array(unlist)] = np.round(ys).astype(int)
    newVelocities[np.where(newVelocities < 0)[0]] = y[-1]
    newVelocities = newVelocities.astype(int)
    return(newVelocities)
    

#m is the number of hidden states type
def hmmcompose(input_filename,output_filename,min_note,m,tol):
    quarter_note, num, denom, key, measures, time, notes, velocity, song, ind = pre_process(input_filename, min_note).read_process()
    #why only old notes and velocity?
    ''''''
    possibleNotes=np.unique(notes)
    k=len(possibleNotes)
    xNotes=encode(notes,possibleNotes)
    n=len(xNotes)
    iteration,p,pi,b,A=optimize(n,m,k,xNotes,tol)
    newNotes,z=generate(n,pi,b,A,code=possibleNotes)
    newVelocities=find_vel(newNotes,velocity)
    
    song.iloc[ind, 1] = time
    song.iloc[ind, 4] = newNotes
    song.iloc[ind, 5] = newVelocities
    song.iloc[ind[np.where(newVelocities !=0)], 2] = ' Note_on_c'
    song.iloc[ind[np.where(newVelocities ==0)], 2] = ' Note_off_c'
    song.to_csv(output_filename, header = None, index = False)
    
