## Annie Bonde
## Class for PCA classifier

import numpy as np


class PCA:
    def __init__(self, W_size):
        # number of components (eigenvectors)
        self.W_size = W_size
        self.mean = None
        # components (eigenvectors)
        self.W = None

    def fit(self, X):
        # subtracting sample mean from original data
        self.mean = np.mean(X, axis=0)
        X_m = X - self.mean

        # covariance matrix
        cov_matrix = np.cov(X_m.T)

        # get eigenvectors and eigenvalues of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # sort in descending order
        idx = eigenvalues.argsort()[::-1]
        self.eigenvalues = eigenvalues[idx]
        self.eigenvectors = eigenvectors[:, idx]

        # select the top w_size eigenvectors
        self.W = self.eigenvectors[:, :self.W_size]

    def transform(self, X):
        # subtract mean
        X_m = X - self.mean

        # project on to top w_size eigenvectors
        # something is funky with my W matrix, cant do transpose here for it to work
        return np.dot(X_m, self.W)

    def back_project(self, Z):
        # dot W.T with Z to bring back to original space
        X = np.dot(Z, self.W[:, :Z.shape[1]].T)
        # add mean back
        X = X + self.mean

        return X
