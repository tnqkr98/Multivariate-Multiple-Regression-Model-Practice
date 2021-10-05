import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns


sample_data = pd.read_csv("sample.csv")
print(sample_data.info(verbose=True, show_counts=True))


def remove_outliers(df, column_name, lower, upper):
    removed_outliers = df[column_name].between(df[column_name].quantile(lower), df[column_name].quantile(upper))

    print(str(df[column_name][removed_outliers].size) + "/" + str(sample_data[column_name].size) + " data points remain.")

    index_names = df[~removed_outliers].index
    return df.drop(index_names)


def PlotMultiplePie(df, categorical_features=None, dropna=False):
    # set a threshold of 30 unique variables, more than 50 can lead to ugly pie charts
    threshold = 30

    # if user did not set categorical_features
    if categorical_features == None:
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


PlotMultiplePie(sample_data)

sample_data = remove_outliers(sample_data, "co2", 0.1, 0.9)
print(sample_data)

