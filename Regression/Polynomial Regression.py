#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 20:10:22 2020

@author: raghavchugh
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("../data/Salary_Data.csv")
dataset = dataset.dropna()

X = dataset.iloc[:,0].values
y = dataset.iloc[:,1].values

X = X.reshape(-1,1)

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
regressor = LinearRegression()
regressor.fit(X_poly, y)

y_predict = regressor.predict(X_poly)

plt.scatter(X, y, color = "blue")
plt.plot(X,y_predict,color="green")

