import numpy as np
import numpy.linalg as LA
import sklearn
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random

class Smote(object):

    def __init__(self, minority_samples):
        self.samples = minority_samples
        pass

    def oversample(self, N: int = 500, k: int = 5):
        N = N//100
        numsamples, numattrs = self.samples.shape
        synthetic = np.zeros( (N*numsamples, numattrs))
        
        matrix=[]
        indices = k_NN(self.samples)
        newindex = 0
        for i in range(numsamples):
            while N > 0:
                nn = random.randint(0, k)
                for attr in range(numattrs):
                    dif = self.samples[indices[i, nn], attr] -self.samples[i, attr]
                    gap = random.uniform(0,1)
                    synthetic[newindex,attr] = self.samples[i, attr] + gap*dif
                newindex += 1
                N -= 1
        return synthetic
        #for m in range(len(indices)):
        #    t=self.samples[indices[m]]
        #    newt=pd.DataFrame(t)
        #    matrix.append([])
        #    for j in range(len(newt.columns)):
        #        matrix[m].append(random.choice(newt[j]))
        return matrix

def k_NN(X: np.ndarray, k: int = 5):
    indices_matrix = np.zeros((X.shape[0], k))
    for idx, row in enumerate(X):
        dist = LA.norm(X-row, axis=1)
        indices = np.argsort(dist)[:k]
        indices_matrix[idx,:] = indices 
    return indices_matrix.astype(int)  

raw_data = pd.read_csv("data.csv")
bankrupt_pd = raw_data["Bankrupt?"]
features_pd = raw_data.drop(["Bankrupt?"], axis=1)
bankrupt = bankrupt_pd.to_numpy()

minority = features_pd[bankrupt_pd == 1]
smote = Smote(minority.to_numpy())
smote.oversample()
