import numpy as np
import sklearn
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class Smote(object):

    def __init__(self, minority_samples: np.ndarray):
        self.samples = minority_samples
        pass

    def oversample(self, T: int, N: int = 500, k: int = 5):
        N = N//100
        numattrs = self.samples.shape[1]
        synthetic = np.zeros( (N*T, numattrs))

        pass


raw_data = pd.read_csv("data.csv")
bankrupt_pd = raw_data["Bankrupt?"]
features_pd = raw_data.drop(["Bankrupt?"], axis=1)
bankrupt = bankrupt_pd.to_numpy()
    