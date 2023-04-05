## Annie Bonde
## Class for KNN classifier
## Used in problems 2 and 3

import numpy as np
from scipy.stats import mode

class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        # don't need to do anything to fit data
        self.X = X
        self.y = y

    def predict(self, test):
        #using euclidean distances
        dists = self.euc_dist(test)
        y_pred = np.zeros(test.shape[0])
        for i in range(test.shape[0]):
            #sort distances in descending order, get class labels
            idx = np.argsort(dists[i, :])[:self.k]
            labels = self.y[idx]
            # assign most most common class among nearest neighbors
            y_pred[i] = mode(labels)[0][0]
        return y_pred

    def euc_dist(self, X):
        # dist from test point X to all other points
        dists = np.zeros((X.shape[0], self.X.shape[0]))
        for i in range(X.shape[0]):
            dists[i, :] = np.sqrt(np.sum((self.X - X[i, :]) ** 2, axis=1))
        return dists


