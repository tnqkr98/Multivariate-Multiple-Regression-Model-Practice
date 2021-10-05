from sklearn.datasets import make_regression
from sklearn.model_selection import RepeatedKFold
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error , r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import pandas as pd

train_data = pd.read_csv('data/train.csv')
validation_data = pd.read_csv('data/test.csv')
train_data = train_data.dropna(axis=0)                  # 결측치 제거

print(train_data.info(verbose=True, show_counts=True))

x_train, y_train = train_data['x'], train_data['y']
x_val, y_val = validation_data['x'], validation_data['y']

x_train = np.array(x_train).reshape(-1, 1)
y_train = np.array(y_train).reshape(-1, 1)
x_val = np.array(x_val).reshape(-1, 1)
y_val = np.array(y_val).reshape(-1, 1)

# Linear Regression
model1 = LinearRegression()
model1.fit(x_train, y_train)
y_preds = model1.predict(x_val)

rmse = np.sqrt(mean_squared_error(y_val, y_preds))
print(f"Model 1 - RMSE : {rmse: .4f}")

# plt.plot(x_train, y_train, 'o')
# plt.plot(x_train, model1.predict(x_train))
# plt.show()

# Decision Tree
model2 = DecisionTreeRegressor()
model2.fit(x_train, y_train)
y_preds = model2.predict(x_val)

rmse = np.sqrt(mean_squared_error(y_val, y_preds))
print(f"Model 2 - RMSE : {rmse: .4f}")

# K-Nearest Neighbors
model3 = KNeighborsRegressor()
model3.fit(x_train, y_train)
y_preds = model3.predict(x_val)

rmse = np.sqrt(mean_squared_error(y_val, y_preds))
print(f"Model 3 - RMSE : {rmse: .4f}")
print(x_val[0][0])
print(type(x_val))
print(f"val = {x_val[1][0]}, truth :{y_val[1][0]}, predict:{model1.predict(x_val)[1][0] }")
print(f"val = {x_val[1][0]}, truth :{y_val[1][0]}, predict:{model2.predict(x_val)[1][0] }")
print(f"val = {x_val[1][0]}, truth :{y_val[1][0]}, predict:{model3.predict(x_val)[1][0] }")
# k Fold