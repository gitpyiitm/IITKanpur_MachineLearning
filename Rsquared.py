# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("data/Position_Salaries.csv")
print(dataset)

X = dataset.iloc[:,1].values
X=X.reshape(-1,1)

Y = dataset.iloc[:,2].values
print(Y)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X,Y)
y_pred = regressor.predict(X)

from sklearn.metrics import r2_score
print(r2_score(Y, y_pred))

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=50, random_state=0)
regressor.fit(X,Y)
y_pred = regressor.predict(X)
print(r2_score(Y, y_pred))

from sklearn.preprocessing import PolynomialFeatures

regressor = LinearRegression()
polyreg = PolynomialFeatures(degree=2)
X = polyreg.fit_transform(X)
regressor.fit(X,Y)
y_pred = regressor.predict(X)
print(r2_score(Y, y_pred))