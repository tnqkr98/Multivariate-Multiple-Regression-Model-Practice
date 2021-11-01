import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.multioutput import RegressorChain
# from bayes_opt import BayesianOptimization
import shap

sample_data = pd.read_csv("sample.csv")
# print(sample_data.info(verbose=True, show_counts=True))


def remove_outliers(df, column_name, lower, upper):
    removed_outliers = df[column_name].between(df[column_name].quantile(lower), df[column_name].quantile(upper))

    print(
        str(df[column_name][removed_outliers].size) + "/" + str(sample_data[column_name].size) + " data points remain.")

    index_names = df[~removed_outliers].index
    return df.drop(index_names)


def PlotMultiplePie(df, categorical_features=None, dropna=False):
    # set a threshold of 30 unique variables, more than 50 can lead to ugly pie charts
    threshold = 30

    # if user did not set categorical_features
    if categorical_features is None:
        categorical_features = df.select_dtypes(['object', 'category']).columns.to_list()

    print("The Categorical Features are:", categorical_features)

    # loop through the list of categorical_features
    for cat_feature in categorical_features:
        num_unique = df[cat_feature].nunique(dropna=dropna)
        num_missing = df[cat_feature].isna().sum()
        # prints pie chart and info if unique values below threshold
        if num_unique <= threshold:
            print('Pie Chart for: ', cat_feature)
            print('Number of Unique Values: ', num_unique)
            print('Number of Missing Values: ', num_missing)
            fig = px.pie(df[cat_feature].value_counts(dropna=dropna), values=cat_feature,
                         names=df[cat_feature].value_counts(dropna=dropna).index, title=cat_feature, template='ggplot2')
            fig.show()
        else:
            print('Pie Chart for ', cat_feature, ' is unavailable due high number of Unique Values ')
            print('Number of Unique Values: ', num_unique)
            print('Number of Missing Values: ', num_missing)
            print('\n')


def evaluateRegressor(true, predicted, message="    Test Set"):
    MSE = mean_squared_error(true, predicted, squared=True)
    MAE = mean_absolute_error(true, predicted)
    RMSE = mean_squared_error(true, predicted, squared=False)
    R_squared = r2_score(true, predicted)

    print(message)
    print("MSE :", MSE)
    print("MAE :", MAE)
    print("RMSE :", RMSE)
    print("R-Squared :", R_squared)


def predictBestEnv(model, time):
    result_env = [0, 0, 0, 0, 0]
    for i in range(time):
        pre = model.predict()


# sns.countplot(sample_data['age'])
# PlotMultiplePie(sample_data)

sample_data = pd.read_csv("sample.csv")
sample_data = remove_outliers(sample_data, "co2", 0.05, 0.95)
idx_zero_temp = sample_data[sample_data['temp'] == 0].index
sample_data = sample_data.drop(idx_zero_temp)
sample_data = pd.get_dummies(sample_data)                       # Embedding

x_data = sample_data.iloc[:, 5:]
y_data = sample_data.iloc[:, [0, 1, 2, 3, 4]]
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=42)

print(x_data.shape)

sc = MaxAbsScaler()
x_train_scaled = sc.fit_transform(x_train)
x_test_scaled = sc.transform(x_test)

y_train_scaled = sc.fit_transform(y_train)
y_test_scaled = sc.transform(y_test)

x_train_scaled = np.array(x_train_scaled, dtype=np.float32)
y_train_scaled = np.array(y_train_scaled, dtype=np.float32)
x_test_scaled = np.array(x_test_scaled, dtype=np.float32)
y_test_scaled = np.array(y_test_scaled, dtype=np.float32)

inputs = x_train_scaled
targets = y_train_scaled

test_inputs = x_test_scaled
test_targets = y_test_scaled



# Initialize Model

print("Random Forest Regressor")
RFRegModel = RandomForestRegressor(random_state=0).fit(inputs, targets)
predict_train_y = RFRegModel.predict(inputs)
evaluateRegressor(targets, predict_train_y, "    Training Set")
predict_valid_y = RFRegModel.predict(test_inputs)
evaluateRegressor(test_targets, predict_valid_y)
predict_valid_y = sc.inverse_transform(predict_valid_y)
evaluateRegressor(y_test, predict_valid_y, "     Valid Set")
print("\n")



"""print("Linear Regression")
LinearModel = LinearRegression().fit(inputs, targets)
predict_train_y = LinearModel.predict(inputs)
evaluateRegressor(targets, predict_train_y, "    Training Set")
predict_valid_y = LinearModel.predict(test_inputs)
evaluateRegressor(test_targets, predict_valid_y)
predict_valid_y = sc.inverse_transform(predict_valid_y)
evaluateRegressor(y_test, predict_valid_y, "     Valid Set")
print("\n")

print("K-Nearest Neighbors")
KNNModel = KNeighborsRegressor().fit(inputs, targets)
predict_train_y = KNNModel.predict(inputs)
evaluateRegressor(targets, predict_train_y, "    Training Set")
predict_valid_y = KNNModel.predict(test_inputs)
evaluateRegressor(test_targets, predict_valid_y)
predict_valid_y = sc.inverse_transform(predict_valid_y)
evaluateRegressor(y_test, predict_valid_y, "     Valid Set")
print("\n")

print("Decision Tree")
DTModel = DecisionTreeRegressor().fit(inputs, targets)
predict_train_y = DTModel.predict(inputs)
evaluateRegressor(targets, predict_train_y, "    Training Set")
predict_valid_y = DTModel.predict(test_inputs)
evaluateRegressor(test_targets, predict_valid_y)
predict_valid_y = sc.inverse_transform(predict_valid_y)
evaluateRegressor(y_test, predict_valid_y, "     Valid Set")
print("\n")

# ToDo : SVR (Multioutput), K-Fold Cross Validation

print("Support Vector Machine - Direct Multioutput")
model = LinearSVR(C=0.1, random_state=1,max_iter=10000000)
model = MultiOutputRegressor(model)
model.fit(inputs, targets)
predict_train_y = model.predict(inputs)
evaluateRegressor(targets, predict_train_y, "    Training Set")
predict_valid_y = model.predict(test_inputs)
evaluateRegressor(test_targets, predict_valid_y)
predict_valid_y = sc.inverse_transform(predict_valid_y)
evaluateRegressor(y_test, predict_valid_y,"    Valid Set")


print("Support Vector Machine - Chained Multioutput")
model = LinearSVR(C=0.1, random_state=1,max_iter=10000000)
model = RegressorChain(model)
model.fit(inputs, targets)
predict_train_y = model.predict(inputs)
evaluateRegressor(targets, predict_train_y, "    Training Set")
predict_valid_y = model.predict(test_inputs)
evaluateRegressor(test_targets, predict_valid_y)
predict_valid_y = sc.inverse_transform(predict_valid_y)
evaluateRegressor(y_test, predict_valid_y,"    Valid Set")"""