import csv
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


raw_data = pd.read_csv("data.csv")


def remove_outlier(feature, name, data):
    q1 = np.percentile(feature, 25)
    q3 = np.percentile(feature, 75)
    iqr = q3-q1
    cut_off = iqr*1.5
    lower_limit = q1-cut_off
    upper_limit = q3+cut_off
    for instance in feature:
        if instance<lower_limit or instance>upper_limit:
            print(instance)
            data = data.drop(data[(data[name] > upper_limit) | (data[name] < lower_limit)].index)
    return data

    #for col in features_pd:
clean_data = remove_outlier(raw_data[" Net worth/Assets"], str(" Net worth/Assets"), raw_data)