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

from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()
X = dataset.data
Y = dataset.target

pX = preprocessing.normalize(X)
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

from sklearn.metrics import precision_score, recall_score, f1_score
print('Precision: %.3f' % precision_score(y_true=Y, y_pred=ay_pred))
print('Recall: %.3f' % recall_score(y_true=Y, y_pred=ay_pred))
print('F1: %.3f' % f1_score(y_true=Y, y_pred=ay_pred))
