#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 23:00:23 2019

@author: mayuheng
"""

import scipy.io as sio
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

def Detectionplot():

    data = sio.loadmat('/Users/mayuheng/Desktop/A.mat')    #完成数据的导入
    m = data['A']#将其与m数组形成对应关系
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')   #此处因为是要和其他的图像一起展示，用的add_subplot，如果只是展示一幅图的话，可以用subplot即可

    x = np.arange(20)
    y = np.arange(20)
    x,y=np.meshgrid(x,y)
    x,y=x.ravel(),y.ravel()
    z = m.flatten('F')
   

    #x = x.flatten('F')   #flatten功能具体可从Declaration中看到
    #y = y.flatten('F')
   
#更改柱形图的颜色，这里没有导入第四维信息，可以用z来表示了
    C = []  
    for a in z:
        if a < 0.4:
            C.append('b')
        elif a < 0.6:
            C.append('c')
        elif a < 0.7:
            C.append('m')
        elif a < 0.8:
            C.append('pink')
        elif a > 1:
            C.append('r')
    
#此处dx，dy，dz是决定在3D柱形图中的柱形的长宽高三个变量
    dx = 0.4 * np.ones_like(x)
    dy = 0.4 * np.ones_like(y)
    dz = abs(z)
    z = np.zeros_like(z)
    
#设置三个维度的标签
    ax.set_xlabel('pieces')
    ax.set_ylabel('pieces')
    ax.set_zlabel('MI')
    plt.axis([0, 21, 0, 21])
    
    ax.bar3d(x, y, z, dx, dy, dz)# color=C)
    
    plt.show()

