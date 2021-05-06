import numpy as np
import numpy.linalg as LA

class PCAnalyser():

    def __init__(self, X: np.ndarray, data_centered=False) -> None:
        if data_centered:
            self.X = X
        else:
            self.X = X - np.mean(X, axis=0)
        print(self.X.T.shape)
        #self.Sigma = self.X.T @ self.X
        return

    def analyse(self, k=20):
        eig_vals, eig_vecs = LA.eigh(self.Sigma)

        idx = np.argsort(eig_vals)[::-1]
        eig_vals = eig_vals[idx]
        eig_vecs = eig_vecs[:,idx]
        PCs = eig_vecs[:,0:k]
        return PCs

    def calc_PVE(eigs, m, individual=False):
        m = np.clip(m, 0, len(eigs))
        if individual:
            return eigs[m] / sum(eigs) # PVE(m)
        return sum(eigs[:m+1]) / sum(eigs) # PVE(first m)
