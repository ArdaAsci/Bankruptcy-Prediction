import numpy as np
import random
import pandas as pd

def remove_outlier(feature, name, data):
    q1 = np.percentile(feature, 25)
    q3 = np.percentile(feature, 75)
    iqr = q3-q1
    cut_off = iqr*1.5
    lower_limit = q1-cut_off
    upper_limit = q3+cut_off
    data = data.drop(data[(data[name] > upper_limit) | (data[name] < lower_limit)].index)
    return data


def test_train_split(data: pd.DataFrame, test_ratio):
    if test_ratio > 1 or test_ratio < 0:
        return
    N = data.shape[0]
    test_amount = int(test_ratio*N)
    test_indices = random.sample(range(N), test_amount)
    test_data = data.iloc[test_indices].reset_index(drop=True)
    train_data = data.drop(test_indices).reset_index(drop=True)
    return train_data, test_data

def confusion_matrix(real, pred, show = True, ret = True):
    TP = np.sum(np.logical_and(real == 1, pred == 1))
    TN = np.sum(np.logical_and(real == 0, pred == 0))
    FN = np.sum(np.logical_and(real == 1, pred == 0))
    FP = np.sum(np.logical_and(real == 0, pred == 1))
    matrix = np.array([[TP, FN], [FP, TN]])
    if show:
        print(" \t1\t0 (prediction)")
        print("1\t", matrix[0, 0], "\t", matrix[0, 1], sep="")
        print("0\t", matrix[1, 0], "\t", matrix[1, 1], sep="")
    if ret:
        return matrix
    return

    def to_categorical(x, n_col=None):
        """ One-hot encoding of nominal values """
        if not n_col:
            n_col = np.amax(x) + 1
        one_hot = np.zeros((x.shape[0], n_col))
        one_hot[np.arange(x.shape[0]), x] = 1
        return one_hot