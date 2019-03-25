# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 14:34:54 2019

@author: PhilippeClaude
"""

from sklearn.neighbors import KNeighborsClassifier
import SVM_fct as svm_fct
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def neighborsClass(X, Y, Z, ZY) :       #Même méthodologie que pour la SVM phase init, predict, et calcul performance et matrice de confusion
    print("---------- Calcul nearest neighbors ball tree")
    
    TEST = []
    TEST2 = []
#    nbrsx = NearestNeighbors(n_neighbors = 2,  algorithm = ball_tree).fit(X)
#    nbrsz = NearestNeighbors(n_neighbors = 2,  algorithm = ball_tree).fit(Z)
    
    neigh = KNeighborsClassifier()
    neigh.fit(X, Y) 
    Xpredict = neigh.predict(X)
    Zpredict = neigh.predict(Z)
    
    TEST.append(Zpredict == ZY)
    TEST2.append(Xpredict == Y)
    
    svm_fct.calculPerformance(TEST, TEST2)
    
    print("Matrice de confusion de dev")
    conf_matrix = confusion_matrix(ZY, Zpredict)
    plt.matshow(conf_matrix)
    plt.colorbar()
    plt.show()
    
    print("Matrice de confusion de test")
    conf_matrix = confusion_matrix(Y, Xpredict)
    plt.matshow(conf_matrix)
    plt.colorbar()
    plt.show()
    #print(" PROBA " + neigh.predict_proba(X))