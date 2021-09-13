# multivariate multiple regression using keras and random dataset

from sklearn.datasets import make_regression
from sklearn.model_selection import RepeatedKFold
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt


def get_model(n_inputs, n_outputs):
    model = Sequential()
    model.add(Dense(20, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(n_outputs))
    model.compile(loss='mae', optimizer='adam')
    return model


# 선형 분산 데이터 생성 (in :10 , out : 3)
def get_dataset():
    x, y = make_regression(n_samples=1000, n_features=3, n_informative=5, n_targets=2, random_state=2)
    print(x)
    return x, y


def evaluate_model(x, y):
    results = list()
    n_inputs, n_outputs = x.shape[1], y.shape[1]

    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # enumerate folds
    for train_ix, test_ix in cv.split(x):
        x_train, x_test = x[train_ix], x[test_ix]
        y_train, y_test = y[train_ix], y[test_ix]

        model = get_model(n_inputs, n_outputs)
        model.fit(x_train, y_train, verbose=0, epochs=100)
        mae = model.evaluate(x_test, y_test, verbose=0)

        print('>$.3f'%mae)
        results.append(mae)
    return results


xx, yy = get_dataset()
result = evaluate_model(xx, yy)
print('MAE: %.3f (%.3f' % (np.mean(result), np.std(result)))
