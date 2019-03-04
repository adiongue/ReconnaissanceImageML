#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 08:58:59 2019

@author: phil
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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

Zdist = np.zeros((10, len(Z)))

for cpt_class in range(0, 10):
    w1.append(X[Y == cpt_class]) #créer le tableau X avec le bon nombre de cases
    m1.append(np.mean(w1[cpt_class], axis=0)) # Création du tableau des barycentres des classes (les points sont des images)
    w2.append(Z[ZY == cpt_class]) #créer le tableau X avec le bon nombre de cases
       
cpt = 0 
saClasse = 0

for i in range(0, 10) :
    Zdist[i] = (abs((m1[i] - Z))**2).sum(axis=1)

ZClasse = np.argmin(Zdist, axis=0)

TEST.append(ZClasse == ZY)

performance = (len(sum(TEST)) - sum(TEST).sum())/len(sum(TEST))
print(performance*100)

Xpca = PCA(n_components = 2)
Xpca.fit(X)