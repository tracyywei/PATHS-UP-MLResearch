#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 09:35:43 2021

@author: tracywei
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepalLength', 'sepalWidth', 'petalLength', 'petalWidth', 'class']
dataset = pd.read_csv(url, names=names)
features = dataset.loc[:,['sepalLength', 'sepalWidth', 'petalLength', 'petal-width']]

# Identifying the Possible Number of Clusters in Data

ks = range(1, 6)
inertias = []
for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters = k)
# Fit model to dataset
    model.fit(features)
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

model = KMeans(n_clusters = 3)
model.fit(features)
labels = model.predict(features)

xs = features.sepallength 
ys = features.sepalwidth
plt.scatter(xs, ys, c = labels, alpha = 0.5)
centroids = model.cluster_centers_
centroids_x = centroids[:,0]
centroids_y = centroids[:,1]
scatterplot = plt.scatter(centroids_x, centroids_y, marker = 'D', s = 50)
plt.title('Iris Classification')
plt.xlabel(names[0])
plt.ylabel(names[1])
plt.show()