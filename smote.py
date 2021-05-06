import numpy as np
import numpy.linalg as LA
import random
import kNN

class Smote(object):

    def __init__(self, minority_samples):
        self.samples = minority_samples
        pass

    def oversample(self, N: int = 500, k: int = 5):
        N = N//100
        numsamples, numattrs = self.samples.shape
        synthetic = np.zeros( (N*numsamples, numattrs))
        
        indices = kNN.k_NN(self.samples)
        newindex = 0
        for i in range(numsamples):
            N2 = N
            while N2 > 0:
                nn = random.randint(0, k-1)
                synthetic[newindex,0] = 1
                for attr in range(1, numattrs):
                    dif = self.samples[indices[i, nn], attr] -self.samples[i, attr]
                    gap = random.uniform(0,1)
                    synthetic[newindex,attr] = self.samples[i, attr] + gap*dif
                newindex += 1
                N2 -= 1
        return synthetic


