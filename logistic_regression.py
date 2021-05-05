import numpy as np


class logistic_regression(object):


    def sigmoid(z):
    sig = 1.0 / (1 + np.exp(-z))
    return sig

    def pred(x,b):
        y = np.dot(x,b)
        return sigmoid(y)


    def classify(prediction, boundary=0.5):
        if prediction <= boundary:
            return 0
        else:
            return 1

    def cost_func(features, labels, b):
        prediction = pred(features,b)
        cost = (-labels*np.log(prediction)-(1-labels)*np.log(1-prediction))
        average = cost.sum() / len(features)
        return average

    def gradient_descent(features, labels, b, learning_rate = 0.2, depth = 1000):
        prediction = pred(features, b)
        gradient = np.dot(features.T,  prediction - labels)
        gradient = gradient / len(features)
        gradient = gradient * learning_rate
        b = b - gradient
        for i in range(depth):
            b = gradient_descent(features, labels, b)
        return b
    

    #weights = logistic_regression.gradient_descent()
   # print(logistic_regression.classify(logistic.regression.pred(features, weights)))