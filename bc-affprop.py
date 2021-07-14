#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 10:23:09 2021

@author: tracywei
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()
X = dataset.data
Y = dataset.target

pX = StandardScaler().fit_transform(X)
from sklearn.cluster import AffinityPropagation
ap = AffinityPropagation()

a_pred = ap.fit_predict(pX)

dframe = pd.DataFrame({'predictions': a_pred, 'target': Y})
ct = pd.crosstab(dframe['predictions'], dframe['target'])
print(ct)

y_pred = np.zeros((569,))
for x in ct.index:
    if ct[0][x] < ct[1][x]:
        y_pred[np.where(a_pred==x)]= 1

print('Affinity Propagation accuracy score:', accuracy_score(Y, y_pred))
print('Affinity Propagation confusion matrix:', confusion_matrix(Y, y_pred))

from sklearn.metrics import precision_score, recall_score, f1_score
print('Precision: %.3f' % precision_score(y_true=Y, y_pred=y_pred))
print('Recall: %.3f' % recall_score(y_true=Y, y_pred=y_pred))
print('F1: %.3f' % f1_score(y_true=Y, y_pred=y_pred))

plt.scatter(X[:,0], X[:,1], c = y_pred, alpha=0.5)
plt.title('Affinity Propagation')
plt.xlabel(dataset.feature_names[0])
plt.ylabel(dataset.feature_names[1])
plt.show()

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.scatter(pX[:,0], pX[:,1], c=a_pred, cmap='jet', edgecolor='None', alpha=0.5)
ax1.set_title('Actual clusters')
ax2.scatter(pX[:,0], pX[:,1], c=y_pred, cmap='jet', edgecolor='None', alpha=0.5)
ax2.set_title('Affinity Propagation results')

# Affinity Propagation accuracy score: 0.9507908611599297
# Affinity Propagation confusion matrix: [[203   9]
# [ 19 338]]
# Precision: 0.974
# Recall: 0.947
# F1: 0.960