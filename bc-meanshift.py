#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 10:22:45 2021

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
from sklearn.cluster import MeanShift, estimate_bandwidth
bandwidth = estimate_bandwidth(pX, quantile=0.09)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs=None, max_iter=300)
m_pred = ms.fit_predict(pX)

dframe = pd.DataFrame({'predictions': m_pred, 'target': Y})
ct = pd.crosstab(dframe['predictions'], dframe['target'])
print(ct)

my_pred = np.zeros((569,))
for x in ct.index:
    if ct[0][x] < ct[1][x]:
        my_pred[np.where(m_pred==x)]= 1

print('Mean-shift accuracy score:', accuracy_score(Y, my_pred))
print('Mean-shift confusion matrix:', confusion_matrix(Y, my_pred))

from sklearn.metrics import precision_score, recall_score, f1_score
print('Precision: %.3f' % precision_score(y_true=Y, y_pred=my_pred))
print('Recall: %.3f' % recall_score(y_true=Y, y_pred=my_pred))
print('F1: %.3f' % f1_score(y_true=Y, y_pred=my_pred))


plt.scatter(X[:,0], X[:,1], c = my_pred, alpha = 0.5)
plt.title('Mean-shift clustering')
plt.xlabel(dataset.feature_names[0])
plt.ylabel(dataset.feature_names[1])
plt.show()

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.scatter(pX[:,0], pX[:,1], c=m_pred, cmap='jet', edgecolor='None', alpha=0.5)
ax1.set_title('Actual clusters')
ax2.scatter(pX[:,0], pX[:,1], c=my_pred, cmap='jet', edgecolor='None', alpha=0.5)
ax2.set_title('Mean-shift results')


# Mean-shift accuracy score: 0.9033391915641477
# Mean-shift confusion matrix: [[184  28]
# [ 27 330]]
# Precision: 0.922
# Recall: 0.924
# F1: 0.923