
#====================================#
# AUTEURS: Barry + Lewis + Abalo     #
# USAGE: python exercice2.py         #
#====================================#

# Importation des packages
from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
import seaborn
import pandas
import matplotlib.pyplot as plt
import sys
from numpy import genfromtxt
import math

class Perceptron():
    def __init__(self,features, n_pas):
        self.x = tf.placeholder(tf.float32, (None,features)) # entrée on ne precise pas le nbre de lignes
        self.w = tf.Variable(tf.zeros((features,1))) #w: initialisée par 0
        self.b = tf.Variable(-0.5, tf.float32) #b: -0.5
        self.predict = tf.nn.sigmoid(tf.matmul(self.x, self.w) + self.b) # la sortie (prediction) 

        # Fonction de perte
        self.y = tf.placeholder(tf.float32, (None,)) #sortie souhaitée
        self.loss = tf.reduce_sum(tf.square(self.y - self.predict), 0) #fonction de perte
        self.train_step = tf.train.GradientDescentOptimizer(n_pas).minimize(self.loss) #optimiseur
        self.sess = tf.Session()
    #Fonction d'entrainement
    def entrainer(self, data_x, data_y, nb_it=4000): 
        self.sess.run(tf.global_variables_initializer()) # Initilisation des variable
        for i in range(0, nb_it):
            self.sess.run(self.train_step, {self.x: data_x, self.y: data_y}) # Entrainer le perceptron
    # Fonction de test
    def tester(self,data_x, data_y):
        p = self.sess.run(self.predict, {self.x: data_x}) # Prédire sur l’ensemble d’apprentissage
        nb_total = p.shape[0]
        err = data_y.flatten() - p.flatten()
        err = err * err
        print("\nTaux de correction = ", (nb_total-np.sum(err)), "/", nb_total)
        P = (nb_total-np.sum(err))/nb_total
        err = np.sum(err)   

        # Affichage de l'erreur et la precision
        print("Precision P = ", round(P*100,2),"%")

        # Prendre les valeurs de w et b
        w_value = self.sess.run(self.w) 
        b_value = self.sess.run(self.b)
       
        print("\nW (poids) = ", w_value)
        print("b (biais) = ",b_value)

if __name__ == "__main__":
    # Chemin de la base

    #train = genfromtxt("data/leukemia/ALLAML.trn",delimiter=" ")
    #test = genfromtxt("data/leukemia/ALLAML.tst", delimiter=" ")

    #train = genfromtxt("data/spam/spam.trn",delimiter=" ")
    #test = genfromtxt("data/spam/spam.tst", delimiter=" ")

    train = genfromtxt("data/ovarian/ovarian.trn",delimiter=" ")
    test = genfromtxt("data/ovarian/ovarian.tst", delimiter=" ")


    # Normaliser de 0 à 1 uniquement
    np.maximum(train[:,-1].astype('int'),0,train[:,-1])
    np.maximum(test[:,-1].astype('int'),0,test[:,-1])
    train[:,-1].astype('int')
    test[:,-1].astype('int')
    features = train[:,:-1].shape[1]
    ligne = train[:,:-1].shape[0]
    print(train[:,:-1].shape)
    pas = 0.5 # le pas n 
    perc =  Perceptron(features,pas)
    perc.entrainer(train[:,:-1],train[:,-1],5000) # nombre d'iterations 10000
    perc.tester(test[:,:-1], test[:,-1])
