#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 17:36:55 2021

@author: tracywei
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()
df = dataset.data
df_target = pd.DataFrame(dataset.target, columns=['target'])
df_features = pd.DataFrame(dataset.data, columns = dataset.feature_names)

# k-means clustering
from sklearn.cluster import AffinityPropagation
model = KMeans(n_clusters = 2)
model.fit(df)
predictions = model.predict(df)

dframe = pd.DataFrame({'predictions': predictions, 'target': dataset.target})
ct = pd.crosstab(dframe['predictions'], dframe['target'])
print(ct)

new_pred_labels = np.zeros((569,))
new_pred_labels[np.where(predictions ==0)] = 1
new_pred_labels[np.where(predictions ==1)] = 0

plt.scatter(df[:,0], df[:,1], c = predictions, alpha = 0.5)
plt.title('Actual clusters')
plt.xlabel(dataset.feature_names[0])
plt.ylabel(dataset.feature_names[1])
plt.show()

plt.scatter(df[:,0], df[:,1], c = new_pred_labels, alpha = 0.5)
plt.title('k-means clustering')
plt.xlabel(dataset.feature_names[0])
plt.ylabel(dataset.feature_names[1])
plt.show()

cm = confusion_matrix(dataset.target, new_pred_labels)
print('K-means Clustering accuracy score:')
print(accuracy_score(dataset.target, new_pred_labels))
print(cm)


# Hierarchical clustering
from sklearn.cluster import AgglomerativeClustering 
hc = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage ='ward')
hc_pred = hc.fit_predict(df)
plt.scatter(df[:,0], df[:,1], c = hc_pred, alpha = 0.5)
plt.title('Hierarchical clustering')
plt.xlabel(dataset.feature_names[0])
plt.ylabel(dataset.feature_names[1])
plt.show()
cm = confusion_matrix(dataset.target, hc_pred)
print('Hierarchical clustering accuracy score:')
print(accuracy_score(dataset.target, hc_pred))
print(cm)


# Mean-shift
from sklearn.cluster import MeanShift, estimate_bandwidth
bandwidth = estimate_bandwidth(df, quantile=0.7, n_samples=569)
ms = MeanShift(bandwidth=bandwidth)
ms.fit(df)
labels = ms.labels_
cluster_centers = ms.cluster_centers_
ms_pred = ms.predict(df)
labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

from itertools import cycle
colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(df[my_members, 0], df[my_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
plt.title('Mean-shift clustering')
plt.show()
cm = confusion_matrix(dataset.target, ms_pred)
print('Mean-shift accuracy score:')
print(accuracy_score(dataset.target, ms_pred))
print(cm)


# Affinity Propagation - need help
from sklearn.cluster import AffinityPropagation
ap = AffinityPropagation(preference =-50)
ap.fit(df)
pred = ap.predict(df)
cluster_centers_indices = clustering.cluster_centers_indices_
n_clusters_ = len(cluster_centers_indices)
plt.scatter(df[:,0], df[:,1], c = pred, alpha = 0.5)
plt.title('Affinity Propagation')
plt.xlabel(dataset.feature_names[0])
plt.ylabel(dataset.feature_names[1])
plt.show()
cm = confusion_matrix(dataset.target, pred)
print('Affinity Propagation accuracy score:')
print(accuracy_score(dataset.target, pred))
print(cm)


# DBSCAN -- need help and incomplete
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=0.3, min_samples=5).fit(df)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
xlabels = db.labels_