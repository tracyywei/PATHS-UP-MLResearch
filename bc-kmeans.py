#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 10:21:22 2021

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

from sklearn.metrics import precision_score, recall_score, f1_score
print('Precision: %.3f' % precision_score(y_true=Y, y_pred=y_pred))
print('Recall: %.3f' % recall_score(y_true=Y, y_pred=y_pred))
print('F1: %.3f' % f1_score(y_true=Y, y_pred=y_pred))

plt.scatter(pX[:,0], pX[:,1], c = k_pred, alpha = 0.5)
plt.title('k-means clustering')
plt.xlabel(dataset.feature_names[0])
plt.ylabel(dataset.feature_names[1])
plt.show()