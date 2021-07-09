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
#pX = StandardScaler().fit_transform(X)

# k-means clustering
pX = StandardScaler().fit_transform(X)
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
print(accuracy_score(dataset.target, y_pred))
cm = confusion_matrix(dataset.target, y_pred)
print(cm)

plt.scatter(pX[:,0], pX[:,1], c = k_pred, alpha = 0.5)
plt.title('k-means clustering')
plt.xlabel(dataset.feature_names[0])
plt.ylabel(dataset.feature_names[1])
plt.show()


# Hierarchical clustering
pX = preprocessing.normalize(X)
from sklearn.cluster import AgglomerativeClustering 
hc = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage = 'ward')
h_pred = hc.fit_predict(pX)

dframe = pd.DataFrame({'predictions': h_pred, 'target': Y})
ct = pd.crosstab(dframe['predictions'], dframe['target'])
print(ct)

hy_pred = np.zeros((569,))
hy_pred[np.where(h_pred==0)]= 0
hy_pred[np.where(h_pred==1)]= 1

print('Hierarchical clustering accuracy score:', accuracy_score(Y, hy_pred))
print('Hierarchical clustering confusion matrix:', confusion_matrix(Y, hy_pred))

plt.scatter(X[:,0], X[:,1], c = hy_pred, alpha = 0.5)
plt.title('Hierarchical clustering')
plt.xlabel(dataset.feature_names[0])
plt.ylabel(dataset.feature_names[1])
plt.show()


# Mean-shift
pX = preprocessing.normalize(X)
from sklearn.cluster import MeanShift, estimate_bandwidth
bandwidth = estimate_bandwidth(pX, quantile=0.09)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs=None, max_iter=300)
m_pred = ms.fit_predict(pX)

dframe = pd.DataFrame({'predictions': m_pred, 'target': Y})
ct = pd.crosstab(dframe['predictions'], dframe['target'])
print(ct)

my_pred = np.zeros((569,))
my_pred[np.where(m_pred==0)]= 1
my_pred[np.where(m_pred==2)]= 1
my_pred[np.where(m_pred==5)]= 1
my_pred[np.where(m_pred==6)]= 1

print('Mean-shift accuracy score:', accuracy_score(Y, my_pred))
print('Mean-shift confusion matrix:', confusion_matrix(Y, my_pred))

plt.scatter(X[:,0], X[:,1], c = my_pred, alpha = 0.5)
plt.title('Mean-shift clustering')
plt.xlabel(dataset.feature_names[0])
plt.ylabel(dataset.feature_names[1])
plt.show()

# Affinity Propagation
from sklearn.cluster import AffinityPropagation
ap = AffinityPropagation()
a_pred = ap.fit_predict(pX)

dframe = pd.DataFrame({'predictions': a_pred, 'target': Y})
ct = pd.crosstab(dframe['predictions'], dframe['target'])
print(ct)

plt.scatter(X[:,0], X[:,1], c = a_pred, alpha = 0.5)
plt.title('Affinity Propagation')
plt.xlabel(dataset.feature_names[0])
plt.ylabel(dataset.feature_names[1])
plt.show()

print('Affinity Propagation accuracy score:', accuracy_score(Y, ay_pred))
print('Affinity Propagation confusion matrix:', confusion_matrix(Y, ay_pred))