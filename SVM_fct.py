# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 18:48:41 2019

@author: PhilippeClaude
"""

from sklearn import svm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def calculPerformance(TEST, TEST2) :
    print("Performance Dev")
    performance = (len(sum(TEST)) - sum(TEST).sum())/len(sum(TEST))  #Calcul de performance sum(TEST) permet de transformer les booléens en entier 1 et 0 (plus simple pour le calcul)
    print(performance*100)                                           #Affichage en pourcentage
    
    print("Performance Train")
    performance = (len(sum(TEST2)) - sum(TEST2).sum())/len(sum(TEST2))
    print(performance*100)

def calculSVMLin(X, Y, Z, ZY) :
    print("Calcul SVM Linear")
    TEST = []
    TEST2 = []
                                            #Initialisation de l'algorithme SVM Linear avec les paramètres de bases
    lin = svm.LinearSVC(max_iter=2100)      #On donne une itération max pour éviter que l'algo prenne trop de temps à s'exécuter.
    lin.fit(X, Y)                           #Utilisation de l'algorithme avec Y pour l'entrainement
    Zpredict = lin.predict(Z)               #Utilisation des éléments calculé par la SVM pour prédire notre nouvel Y
    Xpredict = lin.predict(X)
    TEST.append(Zpredict == ZY)
    TEST2.append(Xpredict == Y)
    
    calculPerformance(TEST, TEST2)
    
    print("Matrice de confusion de dev")
    conf_matrix = confusion_matrix(ZY, Zpredict)    #Calcul de la matrice de confusion
    plt.matshow(conf_matrix)    #Affichage de la matrice de confusion
    plt.colorbar()              #Affichage de la barre à droite pour l'échelle de couleur
    plt.show()
    
    print("Matrice de confusion de test")
    conf_matrix = confusion_matrix(Y, Xpredict)
    plt.matshow(conf_matrix)
    plt.colorbar()
    plt.show()
    
def calculSVMSvr(X, Y, Z, ZY) :
    print("Calcul SVM SVR")
    TEST = []
    TEST2 = []
    svr = svm.SVC(kernel='linear', max_iter=2100, gamma='auto')
    svr.fit(X, Y)
    Zpredict = svr.predict(Z)
    Xpredict = svr.predict(X)
    TEST.append(Zpredict == ZY)
    TEST2.append(Xpredict == Y)
    
    calculPerformance(TEST, TEST2)
    
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


def calculSVMRbf(X, Y, Z, ZY) :
    print("Calcul SVM RBF")
    TEST = []
    TEST2 = []
    svr = svm.SVC(max_iter=2100, gamma='auto')
    svr.fit(X, Y)
    Zpredict = svr.predict(Z)
    Xpredict = svr.predict(X)
    TEST.append(Zpredict == ZY)
    TEST2.append(Xpredict == Y)
    
    calculPerformance(TEST, TEST2)
    
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


def calculSVMPoly(X, Y, Z, ZY) :
    print("Calcul SVM Poly")
    TEST = []
    TEST2 = []
    svr = svm.SVC(kernel='poly', max_iter=2100, gamma='auto')
    svr.fit(X, Y)
    Zpredict = svr.predict(Z)
    Xpredict = svr.predict(X)
    TEST.append(Zpredict == ZY)
    TEST2.append(Xpredict == Y)
    calculPerformance(TEST, TEST2)
    
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