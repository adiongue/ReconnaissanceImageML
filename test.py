#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 08:58:59 2019

@author: phil
"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn.decomposition.pca as skl

X = np.load('data/trn_img.npy')
Y = np.load('data/trn_lbl.npy')
Z = np.load('data/dev_img.npy')
ZY = np.load('data/dev_lbl.npy')

img = X[0].reshape(28,28)
plt.imshow(img, plt.cm.gray)
plt.show()


w1 = []
w2 = []
m1 = []
TEST = []

for cpt_class in range(0, 10):
    w1.append(X[Y == cpt_class]) #créer le tableau X avec le bon nombre de cases
    m1.append(np.mean(w1[cpt_class], axis=0))
    
    w2.append(Z[ZY == cpt_class]) #créer le tableau X avec le bon nombre de cases
       
cpt = 0 
saClasse = 0
for point in Z:
    distMin = abs((m1[0] - point))
    saClasse = 0
    i =1
    while i < len(m1):
        dist = abs(m1[i] - point)
        if(dist < distMin):
            distMin = dist
            saClass = i
        i = i+1
        
    TEST.append(saClasse == ZY[cpt])
    cpt = cpt+1     
    
#skl.check_array(w)
