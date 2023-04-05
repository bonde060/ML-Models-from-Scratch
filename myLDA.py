## Annie Bonde
## class for LDA classifier

import numpy as np

class LDA:
    def __init__(self, W_size):
        self.W_size = W_size
        self.W = None

    def fit(self, X, y):
        # get num of classes
        classes = np.unique(y)
        mean = np.mean(X, axis=0)

        # empty within class and between class scatter matrices
        S_w = np.zeros((X.shape[1], X.shape[1]))
        S_b = np.zeros((X.shape[1], X.shape[1]))
        for c in classes:
            # find xs in class c
            X_c = X[np.where(y == c)]
            # class avg
            mean_c = np.mean(X_c, axis=0)
            # using eqns 6.48 and 6.51 from book
            S_w = S_w + np.dot((X_c - mean_c).T, X_c - mean_c)
            S_b = S_b+ X_c.shape[0] * np.outer(mean_c - mean, mean_c - mean)
        # find eigvec and eigvals of Sw^-1 * Sb
        eigenvalues, eigenvectors = np.linalg.eigh(np.dot(np.linalg.pinv(S_w), S_b))
        # sort in descending order, take the first W_size cols
        idx = eigenvalues.argsort()[::-1][:self.W_size]
        self.W = eigenvectors[:, idx]

    def transform(self, X):
        # dot with eig matrix
        return np.dot(X, self.W)