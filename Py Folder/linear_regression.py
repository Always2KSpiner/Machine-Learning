# -*- coding: utf-8 -*-
"""
This is linear regression for salary data 

"""
import matplotlib.pyplot as plt

import pandas as pd

data_set =pd.read_csv("Salary_Data.csv")

X = data_set.iloc[:,:-1].values
Y = data_set.iloc[:,:1].values
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 1/3, random_state = 0 )
from sklearn.linear_model import LinearRegression
simplelinear = LinearRegression()
simplelinear.fit(X_train,Y_train)
y_predict = simplelinear.predict(X_test)

plt.scatter(X_train ,Y_train)
plt.plot(X_train,simplelinear.predict(X_train))
plt.show()