import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import datetime
import pandas_datareader.data as web
from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


import mlflow.sklearn


def acquire_training_data():
    start = datetime.datetime(2019, 7, 1)
    end = datetime.datetime(2019, 9, 30)
    df = web.DataReader("BTC-USD", 'yahoo', start, end)
    return df

def digitize(n):
    if n > 0:
        return 1
    return 0


def rolling_window(a, window):
    """
        Takes np.array 'a' and size 'window' as parameters
        Outputs an np.array with all the ordered sequences of values of 'a' of size 'window'
        e.g. Input: ( np.array([1, 2, 3, 4, 5, 6]), 4 )
             Output: 
                     array([[1, 2, 3, 4],
                           [2, 3, 4, 5],
                           [3, 4, 5, 6]])
    """
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def prepare_training_data(data):

    """
        Return a prepared numpy dataframe
        input : Dataframe with expected schema

    """
    data['Delta'] = data['Close'] - data['Open']
    data['to_predict'] = data['Delta'].apply(lambda d: digitize(d))
    return data

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    #with mlflow.start_run():


    training_data = acquire_training_data()

    prepared_training_data = prepare_training_data(training_data)

    print(len(prepared_training_data))

    btc_mat = prepared_training_data.as_matrix()

    WINDOW_SIZE = 14

    X = rolling_window(btc_mat[:, 7], WINDOW_SIZE)[:-1, :]
    Y = prepared_training_data['to_predict'].as_matrix()[WINDOW_SIZE:]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=4284, stratify=Y)

    clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                           max_depth=None, max_features='auto', max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=None,
                           oob_score=False, random_state=4284, verbose=0,
                           warm_start=False)

    clf.fit(X_train, y_train)


    predicted = clf.predict(X_test)

    mlflow.sklearn.log_model(clf, "model_random_forest")

    print(classification_report(y_test, predicted))

    mlflow.log_metric("precision_label_0", precision_score(y_test, predicted, pos_label=0))
    mlflow.log_metric("recall_label_0", recall_score(y_test, predicted, pos_label=0))
    mlflow.log_metric("f1score_label_0", f1_score(y_test, predicted, pos_label=0))
    mlflow.log_metric("precision_label_1", precision_score(y_test, predicted, pos_label=1))
    mlflow.log_metric("recall_label_1", recall_score(y_test, predicted, pos_label=1))
    mlflow.log_metric("f1score_label_1", f1_score(y_test, predicted, pos_label=1))

