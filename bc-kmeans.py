#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 10:21:22 2021

@author: tracywei
"""

# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# import breast cancer dataset
from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()
X = dataset.data
Y = dataset.target

# preprocessing data
pX = StandardScaler().fit_transform(X)

# initializing model with parameters
from sklearn.cluster import KMeans
model = KMeans(n_clusters=7, init='k-means++', n_init=10, max_iter=300, random_state=1)

# fit model
k_pred = model.fit_predict(pX)

# create crosstab
df = pd.DataFrame({'prediction': k_pred, 'ground-truth': Y})
ct = pd.crosstab(df['prediction'], df['ground-truth'])
print(ct) 

# assign cluster numbers to label numbers
y_pred = np.zeros((569,))
for x in ct.index:
    if ct[0][x] < ct[1][x]:
        y_pred[np.where(k_pred==x)]= 1

# print metrics for model
print('K-means clustering accuracy score:', accuracy_score(Y, y_pred))
print('K-means clustering confusion matrix:', confusion_matrix(Y, y_pred))

from sklearn.metrics import precision_score, recall_score, f1_score
print('Precision: %.3f' % precision_score(y_true=Y, y_pred=y_pred))
print('Recall: %.3f' % recall_score(y_true=Y, y_pred=y_pred))
print('F1: %.3f' % f1_score(y_true=Y, y_pred=y_pred))

# plot
plt.scatter(X[:,0], X[:,1], c = y_pred, alpha = 0.5)
plt.title('K-means Clustering')
plt.xlabel(dataset.feature_names[0])
plt.ylabel(dataset.feature_names[1])
plt.show()

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.scatter(pX[:,0], pX[:,1], c=k_pred, cmap='jet', edgecolor='None', alpha=0.5)
ax1.set_title('Actual clusters')
ax2.scatter(pX[:,0], pX[:,1], c=y_pred, cmap='jet', edgecolor='None', alpha=0.5)
ax2.set_title('KMeans clustering results')

# metrics results
# K-means clustering accuracy score: 0.9332161687170475
# K-means clustering confusion matrix:[[185  27]
# [ 11 346]]
# Precision: 0.928
# Recall: 0.969
# F1: 0.948