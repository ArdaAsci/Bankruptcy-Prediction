import numpy as np
import numpy.linalg as LA

def k_NN(X: np.ndarray, k: int = 5):
    # Performs k-NN for every row of the input X
    indices_matrix = np.zeros((X.shape[0], k))
    for idx, row in enumerate(X):
        dist = LA.norm(X-row, axis=1)
        indices = np.argsort(dist)[:k]
        indices_matrix[idx,:] = indices 
    return indices_matrix.astype(int) 

class k_NN_classifier():

    def __init__(self, X: np.ndarray, Y: np.ndarray) -> None:
        self.X = X
        self.Y = Y
        pass

    def classify(self, data_point: np.ndarray, k = 5):
        dist = LA.norm(self.X - data_point, axis=1)
        indices = np.argsort(dist)[:k]
        neighbor_labels = self.Y[indices]
        return (np.sum(neighbor_labels) / k) >= 0.5 # return 1 if more neighhoring 1's than 0's
