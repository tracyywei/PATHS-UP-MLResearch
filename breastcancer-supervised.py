#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 15:35:34 2021

@author: tracywei
"""


#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()
df = dataset.data
df_target = pd.DataFrame(dataset.target, columns=['target'])
df_features = pd.DataFrame(dataset.data, columns = dataset.feature_names)

scatterplot = plt.scatter(df[:, 0], df[:, 1], c=dataset.target)
plt.title('Tumor Classification')
plt.xlabel(dataset.feature_names[0])
plt.ylabel(dataset.feature_names[1])
plt.legend(handles=scatterplot.legend_elements()[0], labels=['malignant', 'benign'], title="classification")
plt.show()