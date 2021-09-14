# Basic Imports
import numpy as np
import pandas as pd

# Plotting
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

# Preprocessing
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder

# Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ML Models
import lightgbm as lgb
from lightgbm import LGBMRegressor
import xgboost as xg
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.linear_model import LinearRegression
# Model Tuning
#from bayes_opt import BayesianOptimization

# Feature Importance
import shap

# Ignore Warnings
import warnings
warnings.filterwarnings('ignore')

SP_csv = pd.read_csv("data/StudentsPerformance.csv")
SP_csv['average score'] = SP_csv[['math score', 'reading score','writing score']].mean(axis=1)      # average score 추가
print(SP_csv.info(verbose=True, null_counts=True))


def plotMultiplePie(df, categorical_features = None, dropna = False):
    threshold = 30
    if categorical_features is None:
        categorical_features = df.select_dtypes(['object','category']).columns.to_list()
    print("The Categorical Features are : ", categorical_features)

    for cat_feature in categorical_features:     # 선택한 feature 가 없다면 데이터 타입이 object/category 인것만
        num_unique = df[cat_feature].nunique(dropna=dropna)
        num_missing = df[cat_feature].isna().sum()

        if num_unique <= threshold:                 # 범주 종류가 30개 이하인경우(즉 연속형이 아닌 범주형만 -> 당연히 그래야. 연속형은 파이로 플롯못함)
            print('Pie Chart for: ', cat_feature)
            print('Number of Unique Value: ', num_unique)       # 해당 Feature의 범주의 종류
            print('Number of Missing Value: ', num_missing)
            fig = px.pie(df[cat_feature].value_counts(dropna=dropna), values=cat_feature, names=df[cat_feature].value_counts(dropna=dropna).index, title=cat_feature, template='ggplot2')
            fig.show()
        else:
            print('Pie Chart for ',cat_feature,' is unavailable due high number of Unique Values ')
            print('Number of Unique Values: ', num_unique)
            print('Number of Missing Values: ', num_missing)
            print('\n')


#plotMultiplePie(SP_csv)

continuos_features = SP_csv.select_dtypes(['float64', 'int64']).columns.to_list()

for cont_feature in continuos_features:
    plt.figure()
    plt.title(cont_feature)
    ax = sns.distplot(SP_csv[cont_feature])
    plt.show()

