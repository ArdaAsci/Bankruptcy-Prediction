from numba import jit, njit
import numpy as np

@njit(fastmath=True)
def divide_on_feature(X, col_idx, th):
    mask = X[:,col_idx] >= th
    X1 = X[mask]
    X2 = X[~mask]
    valid = X1.size != 0 and X2.size != 0
    return X1, X2, valid


class DecisionNode():
    "Node Struct"
    def __init__(self, col_idx=None, th=None,
                 value=None, tb=None, fb=None):
        self.col_idx = col_idx          
        self.th = th      
        self.value = value              
        self.tb = tb # True Branch
        self.fb = fb # False Branch

class DecisionTree(object):
    def __init__(self, min_split=2, min_entropy=1e-7,
                 max_depth=20):
        self.tree = None
        self.min_split = min_split
        self.min_entropy = min_entropy
        self.max_depth = max_depth

        
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.one_dim = len(y.shape) == 1
        self.tree = self.tree_builder(X, y, 0)

    def tree_builder(self, X, y, current_depth):
        m, n = X.shape
        largest_entropy = 0
        best_criteria = None  
        best_sets = None      
        if len(y.shape) == 1:
            y = np.expand_dims(y, axis=1)
        Xy = np.concatenate((X, y), axis=1)# For splitting rows later


        if m >= self.min_split and current_depth <= self.max_depth:
            # Calculate the entropy for each PC
            for col_idx in range(n):
                # All values of col_idx
                feature_values = np.expand_dims(X[:, col_idx], axis=1)
                sample_size = int(m/5)+23 if m >= 30 else m
                numbers = np.random.default_rng().choice(m, size=int(m/5), replace=False)
                for th in feature_values[numbers,0]:
                    Xy1, Xy2, valid = divide_on_feature(Xy, col_idx, th)

                    if not valid: #Split done
                        continue
                    y1 = Xy1[:, n:]
                    y2 = Xy2[:, n:]

                    entropy = DecisionTree.entropy_calc(y, y1, y2)

                    if entropy > largest_entropy:
                        largest_entropy = entropy
                        best_criteria = {"col_idx": col_idx, "th": th}
                        best_sets = {
                            "leftX": Xy1[:, :n], 
                            "lefty": Xy1[:, n:], 
                            "rightX": Xy2[:, :n],
                            "righty": Xy2[:, n:] 
                            }

        if largest_entropy > self.min_entropy:
            # Build subtrees for the right and left branches
            tb = self.tree_builder(best_sets["leftX"], best_sets["lefty"], current_depth + 1)
            fb = self.tree_builder(best_sets["rightX"], best_sets["righty"], current_depth + 1)
            return DecisionNode(col_idx=best_criteria["col_idx"], th=best_criteria[
                                "th"], tb=tb, fb=fb)
        else:
            leaf_value = np.mean(y)
            return DecisionNode(value=leaf_value)


    @staticmethod
    @njit()
    def entropy_calc(y, y1, y2):
        var = np.var(y)
        var_1 = np.var(y1)
        var_2 = np.var(y2)
        rat1 = y1.size / y.size
        rat2 = y2.size / y.size
        variance_reduction = var - (rat1 * var_1 + rat2 * var_2)
        return variance_reduction

    def predict_value(self, x, tree=None):
        if tree is None:
            tree = self.tree
        if tree.value is not None:
            return tree.value
        feature_value = x[tree.col_idx]

        branch = tree.fb
        if feature_value >= tree.th:
            branch = tree.tb
        return self.predict_value(x, branch)

    def predict(self, X):
        y_pred = [self.predict_value(sample) for sample in X]
        y_pred = np.array(y_pred)
        return y_pred


class CrossEntropy():
    @staticmethod
    @njit(fastmath=True)
    def gradient(y, p):
        return - (y / p) + (1 - y) / (1 - p)


@njit
def sigoid(x):
    return np.exp(x) / (1 + np.exp(x))

class BinaryGradientBoostClassify():

    def __init__(self, tree_count, lr, tree_min_split, tree_min_entropy, tree_max_depth):
        self.tree_count = tree_count
        self.lr = lr
        self.tree_min_split = tree_min_split
        self.tree_min_entropy = tree_min_entropy
        self.tree_max_depth = tree_max_depth

        self.trees = []
        for i in range(self.tree_count):
            self.trees.append(DecisionTree(
                                min_split=self.tree_min_split,
                                min_entropy=self.tree_min_entropy,
                                max_depth=self.tree_max_depth) )


    def fit(self, X: np.ndarray, y: np.ndarray):
        y_hat = np.full(np.shape(y), np.mean(y, axis=0))
        for i in range(self.tree_count):
            gradient = CrossEntropy.gradient(y, y_hat)
            self.trees[i].fit(X, gradient)
            update = self.trees[i].predict(X)
            y_hat -= np.multiply(self.lr, update)


    def predict(self, X: np.ndarray):
        y_hat = np.array([])
        for tree in self.trees:
            update = tree.predict(X)
            if y_hat.all():
                y_hat = -update
            else:
                y_hat -= update
        y_hat = sigoid(y_hat)
        return y_hat > 0.68
