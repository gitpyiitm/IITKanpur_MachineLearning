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

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

X_train = X_train.reshape(-1,1)
X_test = X_test.reshape(-1,1)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_predict = regressor.predict(X_test)
y_pred = regressor.predict([[15]])

plt.scatter(X_test, y_test, color = "blue")
plt.scatter(X_train, y_train, color ="red")
plt.plot(X_test,y_predict,color="green")
