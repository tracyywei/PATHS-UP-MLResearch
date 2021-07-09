#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 10:21:58 2021

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

from sklearn.metrics import precision_score, recall_score, f1_score
print('Precision: %.3f' % precision_score(y_true=Y, y_pred=hy_pred))
print('Recall: %.3f' % recall_score(y_true=Y, y_pred=hy_pred))
print('F1: %.3f' % f1_score(y_true=Y, y_pred=hy_pred))

plt.scatter(X[:,0], X[:,1], c = hy_pred, alpha = 0.5)
plt.title('Hierarchical clustering')
plt.xlabel(dataset.feature_names[0])
plt.ylabel(dataset.feature_names[1])
plt.show()