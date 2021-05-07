import numpy as np
import numpy.linalg as LA

class PCAnalyser():

    def __init__(self, X: np.ndarray, data_centered=False) -> None:
        if data_centered:
            self.X = X
        else:
            self.X = X - np.mean(X, axis=0)
        self.Sigma = self.X.T @ self.X
        self.eigs = np.array([])
        return

    def analyse(self, k = 10):
        if k > self.Sigma.shape[0]: return

        eig_vals, eig_vecs = LA.eigh(self.Sigma)
        idx = np.argsort(eig_vals)[::-1]
        self.eigs = eig_vals[idx]
        eig_vecs = eig_vecs[:,idx]
        PCs = eig_vecs[:,0:k]

        return self.eigs, PCs

    def calc_PVE(self, m=10, individual=False):
        m = np.clip(m, 0, len(self.eigs))
        if individual:
            return self.eigs[m] / sum(self.eigs) # PVE(m)
        return sum(self.eigs[:m+1]) / sum(self.eigs) # PVE(first m)