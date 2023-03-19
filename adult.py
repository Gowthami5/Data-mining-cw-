# import section
import csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier


# Part 1: Decision Trees with Categorical Attributes

# Return a pandas dataframe containing the data set that needs to be extracted from the data_file.
# data_file will be populated with the string 'adult.csv'.
def read_csv_1(data_file):
    # read csv data and drop column 'fnlwgt'
    df = pd.read_csv(data_file, index_col=False, skipinitialspace=True).drop(['fnlwgt'], axis=1)
    # replace null values ('?') in data
    df.replace(to_replace=[r' ?', r'?', r'? ', r' ', r''], value=[np.nan, np.nan, np.nan, np.nan, np.nan], regex=False,
               inplace=True)

    return df


# Return the number of rows in the pandas dataframe df.
def num_rows(df):
    r = df.shape[0]
    return r


# Return a list with the column names in the pandas dataframe df.
def column_names(df):
    # create the list of col names
    data_colname = [col_name for col_name in df.columns]
    return data_colname


# Return the number of missing values in the pandas dataframe df.
def missing_values(df):
    # calculate the sum of null (missing) values in dataframe
    miss_val = sum(df.isnull().sum())
    return miss_val


# Return a list with the columns names containing at least one missing value in the pandas dataframe df.
def columns_with_missing_values(df):
    # count of column of null (missing) values in dataframe
    data_isnull = df.isnull().sum()
    # select only the column of null (missing) values in dataframe and convert to list
    data_col_isnull = data_isnull[data_isnull > 0].index.values.tolist()
    return data_col_isnull


# Return the percentage of instances corresponding to persons whose education level is
# Bachelors or Masters, by rounding to the third decimal digit,
# in the pandas dataframe df containing the data set in the adult.csv file.
# For example, if the percentage is 0.21547%, then the function should return 0.216.
def bachelors_masters_percentage(df):
    count_HEdu = None
    try:
        dictEdu = df.education.value_counts(normalize=True).to_dict()
        count_HEdu = dictEdu['Bachelors'] + dictEdu['Masters']
        count_HEdu = round(count_HEdu, 3)
    except Exception as x:
        print('No variable name \'education\' in data')

    return count_HEdu


# Return a pandas dataframe (new copy) obtained from the pandas dataframe df
# by removing all instances with at least one missing value.
def data_frame_without_missing_values(df):
    df = df.dropna()
    return df


# Return a pandas dataframe (new copy) from the pandas dataframe df
# by converting the df categorical attributes to numeric using one-hot encoding.
# The function should not encode the target attribute, and the function's output
# should not contain the target attribute.
def one_hot_encoding(df):

    # seperate one-hot categorical names within columns
    columns = [column for column in df if ((df[column].dtype == object) & (column.lower().strip() != 'class'))]
    # get one hot encodings
    df_dummy = pd.get_dummies(df[columns])
    # remove processed class column and add back the old version of data
    df_class = df[df.columns[-1]]
    df = df.join(df_dummy).drop(columns, axis=1).drop(df.columns[-1], axis=1)
    df = df.join(df_class)

    return df


# Return a pandas series (new copy), from the pandas dataframe df,
# containing only one column with the labels of the df instances
# converted to numeric using label encoding. 
def label_encoding(df):
    # creating instance of labelencoder
    labelencoder = LabelEncoder()
    # encode the class column
    df[df.columns[-1]] = labelencoder.fit_transform(df[df.columns[-1]])

    return df


# Given a training set X_train containing the input attribute values
# and labels y_train for the training instances,
# build a decision tree and use it to predict labels for X_train. 
# Return a pandas series with the predicted values. 
def dt_predict(X_train, y_train):
    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier()
    # Train Decision Tree Classifer
    clf = clf.fit(X_train, y_train)
    # Predict the response for train dataset
    y_pred = clf.predict(X_train)

    return pd.Series(y_pred)


# Given a pandas series y_pred with the predicted labels and a pandas series y_true with the true labels,
# compute the error rate of the classifier that produced y_pred.  
def dt_error_rate(y_pred, y_true):
    # error rate = (ypred != ytrue) / Total number of observation
    err_rate = sum(y_pred.to_numpy(dtype=float) != y_true.to_numpy(dtype=float)) / y_true.shape[0]
    return round(err_rate, 3)


# test Script
if __name__ == "__main__":
    # 1. [10 points] Read the data set and compute: (a) the number of instances, (b) a list with the
    # attribute names, (c) the number of missing values, (d) a list of the attribute names with at
    # least one missing value, and (e) the percentage of instances corresponding to individuals whose
    # education level is Bachelors or Masters (real number rounded to the first decimal digit).

    # read adult.csv
    df = read_csv_1('adult.csv')

    # 1.a.
    r = num_rows(df)
    print('1.a. num_rows:\n', r)
    # 1.b
    listname = column_names(df)
    print('1.b. column_names:\n', listname)
    # 1.c
    num_miss = missing_values(df)
    print('1.c. number of missing values:\n', str(num_miss))
    # 1.d
    data_isnull = columns_with_missing_values(df)
    print('1.d. columns_with_missing_values:\n', data_isnull)
    # 1.e
    uniq_n = bachelors_masters_percentage(df)
    print('1.e. bachelors_masters_percentage:\n', uniq_n)

    # 2.[10 points] Drop all instances with missing values. Convert all attributes (except the class) to
    # numeric using one-hot encoding. Name the new columns using attribute values from the original
    # data set. Next, convert the class values to numeric with label encoding.
    df = data_frame_without_missing_values(df)
    # 2.a
    data_isnull = columns_with_missing_values(df)
    print('2.a. columns_with_missing_values after dropping NAN:\n', data_isnull)
    # 2.b
    df = one_hot_encoding(df)
    # 2.c
    df = label_encoding(df)

    # 3.[10 points] Build a decision tree and classify each instance to one of the <= 50K and > 50K
    # categories. Compute the training error rate of the resulting tree.
    feature_cols = [col_name for col_name in df.columns.difference([df.columns[-1]])]
    X_train = df[feature_cols]  # Features
    y_train = df['class']  # Target variable
    y_pred = dt_predict(X_train, y_train)
    print('3.a. predicted series: \n', y_pred)
    error_rate = dt_error_rate(y_pred, y_train)
    print('3.b. predicted error_rate: \n', error_rate)

    # df.info()
    df.head()
