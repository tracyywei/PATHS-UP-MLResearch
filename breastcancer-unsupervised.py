#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 21:28:20 2021

@author: tracywei
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()
df = dataset.data
df_target = pd.DataFrame(dataset.target, columns=['target'])
df_features = pd.DataFrame(dataset.data, columns = dataset.feature_names)


# Identifying the Possible Number of Clusters in Data

ks = range(1, 6)
inertias = []
for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters = k)
# Fit model to dataset
    model.fit(df)
# Append the inertia to the list of inertias
    inertias.append(model.inertia_)
# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()
# 2 is the ideal k (number of clusters)


# Checking the Quality of Clusters

model = KMeans(n_clusters = 2)
model.fit(df)
labels = model.predict(df)

xs = df[:,0] 
ys = df[:,1] 
plt.scatter(xs, ys, c = labels, alpha = 0.5)
centroids = model.cluster_centers_
centroids_x = centroids[:,0]
centroids_y = centroids[:,1]
scatterplot = plt.scatter(centroids_x, centroids_y, marker = 'D', s = 50)
plt.title('Tumor Classification')
plt.xlabel(dataset.feature_names[0])
plt.ylabel(dataset.feature_names[1])
plt.show()

# Creating a dataframe of predicted labels and target (classification)
dframe = pd.DataFrame({'labels': labels, 'target': dataset.target})
# Crosstab evaluation to verify the quality of our clustering
ct = pd.crosstab(dframe['labels'], dframe['target'])
print(ct)

# Feature Scaling and Normalization

scaler = StandardScaler()
kmeans = KMeans(n_clusters= 2)
pipeline = Pipeline([('Scaler',scaler), ('KMeans',kmeans)])
pipeline.fit(df)
labels2 = pipeline.predict(df)
df2 = pd.DataFrame({'labels': labels2, 'target': dataset.target})
ct2 = pd.crosstab(df2['labels'], df2['target'])
print(ct2)