import numpy as np


class logistic_regression(object):

    def __init__(self, size) -> None:
        self.b = np.random.rand(size,1)
        pass

    def sigmoid(z:float):
        sig = 1.0 / (1 + np.exp(-z))
        return sig

    def pred(features, weight):
        y = features@weight
        return logistic_regression.sigmoid(y)


    def classify(prediction, boundary=0.5):
        return prediction > boundary

    def cost_func(self, features:np.ndarray, labels):
        prediction = logistic_regression.pred(features,self.b)
        cost = (-labels*np.log(prediction)-(1-labels)*np.log(1-prediction))
        average = cost.sum() / len(features)
        return average

    def gradient_descent( features:np.ndarray, labels, weight, learning_rate = 0.02):
        prediction = logistic_regression.pred(features, weight)
        gradient = features.T@ (prediction - labels)
        gradient = gradient / len(features)
        gradient = gradient * learning_rate
        weight = weight - gradient
        return weight
    

    #weights = logistic_regression.gradient_descent()
   # print(logistic_regression.classify(logistic.regression.pred(features, weights)))