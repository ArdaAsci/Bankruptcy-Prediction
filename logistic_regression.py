import numpy as np

class Logistic_regression(object):

    def __init__(self, size) -> None:
        selfb = np.random.rand(size,1)
        pass
    
    
    def sigmoid(z):
        sig = 1.0 / (1 + np.exp(-z))
        return sig

    def pred(features, weight):
        y = features@weight
        return Logistic_regression.sigmoid(y)


    def classify(prediction, boundary=0.50):
        return prediction > boundary


    def cost_func(self, features:np.ndarray, labels):
        prediction = Logistic_regression.pred(features,self.b)
        cost = (-labels*np.log(prediction)-(1-labels)*np.log(1-prediction))
        average = cost.sum() / len(features)
        return average

    def gradient_descent( features:np.ndarray, labels, weight, learning_rate = 0.03):
        prediction = Logistic_regression.pred(features, weight)
        mid = prediction - labels
        gradient = features.T@ mid
        gradient = gradient / len(features)
        gradient = gradient * learning_rate
        weight = weight - gradient
        return weight