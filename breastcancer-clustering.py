#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 17:36:55 2021

@author: tracywei
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()
X = dataset.data
Y = dataset.target

#pX = preprocessing.normalize(X)
pX = StandardScaler().fit_transform(X)

# k-means clustering
from sklearn.cluster import KMeans
model = KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300)
k_pred = model.fit_predict(pX)

df = pd.DataFrame({'prediction': k_pred, 'ground-truth': Y})
ct = pd.crosstab(df['prediction'], df['ground-truth'])
print(ct) 

y_pred = np.zeros((569,))
y_pred[np.where(k_pred==0)]= 1
y_pred[np.where(k_pred==1)]= 0

print('K-means Clustering accuracy score:')
print(accuracy_score(Y, y_pred))
cm = confusion_matrix(Y, y_pred)
print(cm)

plt.scatter(pX[:,0], pX[:,1], c = y_pred, alpha = 0.5)
plt.title('k-means clustering')
plt.xlabel(dataset.feature_names[0])
plt.ylabel(dataset.feature_names[1])
plt.show()


# Hierarchical clustering

from sklearn.cluster import AgglomerativeClustering 
hc = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage = 'ward')
h_pred = hc.fit_predict(pX)

dframe = pd.DataFrame({'predictions': h_pred, 'target': Y})
ct = pd.crosstab(dframe['predictions'], dframe['target'])
print(ct)

y_pred = np.zeros((569,))
y_pred[np.where(h_pred==0)]= 0
y_pred[np.where(h_pred==1)]= 1

print('Hierarchical clustering accuracy score:')
print(accuracy_score(Y, h_pred))
cm = confusion_matrix(Y, h_pred)
print(cm)

plt.scatter(X[:,0], X[:,1], c = h_pred, alpha = 0.5)
plt.title('Hierarchical clustering')
plt.xlabel(dataset.feature_names[0])
plt.ylabel(dataset.feature_names[1])
plt.show()


# Mean-shift
from sklearn.cluster import MeanShift
ms = MeanShift()
m_pred = ms.fit_predict(pX)

dframe = pd.DataFrame({'predictions': m_pred, 'target': Y})
ct = pd.crosstab(dframe['predictions'], dframe['target'])
print(ct)

my_pred = np.zeros((569,))
my_pred[np.where(m_pred==0)]= 1
my_pred[np.where(m_pred==11)]= 1
my_pred[np.where(m_pred==12)]= 1
my_pred[np.where(m_pred==13)]= 1

print('Mean-shift accuracy score:')
print(accuracy_score(Y, my_pred))
cm = confusion_matrix(Y, my_pred)
print(cm)

plt.scatter(X[:,0], X[:,1], c = my_pred, alpha = 0.5)
plt.title('Mean-shift clustering')
plt.xlabel(dataset.feature_names[0])
plt.ylabel(dataset.feature_names[1])
plt.show()
