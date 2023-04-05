##
# Annie Bonde
# 3/26/2023
# Class for EM model


import numpy as np
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans

class EM:
    def __init__(self, data, k):
        self.data = data
        self.k = k
        kmean = KMeans(n_clusters = self.k, max_iter = 1).fit(self.data)
        self.means = kmean.cluster_centers_
        self.cov = np.array([np.diag(np.var(self.data, axis=0))] * k)
        self.weights = np.array([1 / k] * k)
        self.e_vals = np.zeros((data.shape[0], k), dtype=np.float64)

    def fit(self, max_thresh = 100):
        ll = self.log_likelihood()
        self.logs = np.zeros((max_thresh+1, 2))
        self.logs[0, :] = [ll, ll]

        for i in range(max_thresh):
            self.e_step()
            ll_e = self.log_likelihood()
            self.m_step()
            ll_m = self.log_likelihood()
            self.logs[i+1] = [ll_e, ll_m]
            if abs(ll - ll_m) < 1e-4:
                break
            ll = ll_m
        param = {'h': self.e_vals, 'm': self.means, 'Q': self.logs[:, 1]}
        return param

    def fit2(self, max_thresh = 100):
        ll = self.log_likelihood()
        self.logs = np.zeros((max_thresh+1, 2))
        self.logs[0, :] = [ll, ll]

        for i in range(max_thresh):
            self.e_step()
            ll_e = self.log_likelihood()
            self.m_step2()
            ll_m = self.log_likelihood()
            self.logs[i+1] = [ll_e, ll_m]
            if abs(ll - ll_m) < 1e-4:
                break
            ll = ll_m
        param = {'h': self.e_vals, 'm': self.means, 'Q': self.logs[:, 1]}
        return param

    def predict(self, ax, dims):
        n = self.e_vals.shape[0]
        labels = np.zeros((n, 3))
        for i in range(n):
            labels[i, :] = self.means[np.argmax(self.e_vals[i])]

        ax.imshow(labels.reshape(dims))
        ax.set_title(f"Compressed Image, k={self.k}")
        return labels

    def plot_logs(self, ax):
        ax.plot(self.logs[:, 0], label = "After E-step")
        ax.plot(self.logs[:, 1], label = "After M-step")
        ax.set_title("Complete Log-Likelihood (Q), k = "+str(self.k))
        ax.set_xlabel("Iteration")
        ax.legend()

    ##############################################################################

    def update_means(self):
        n = self.data.shape[0]
        counts = np.sum(self.e_vals, axis=0)
        for j in range(self.k):
            weighted_sum = np.sum(self.e_vals[:, j, np.newaxis] * self.data, axis=0)
            self.means[j] = weighted_sum / counts[j]

    def update_cov(self):
        n, d = self.data.shape
        counts = np.sum(self.e_vals, axis=0)
        c = 0
        for i in range(self.k):
            cov_diff = self.data - self.means[i]
            for x in range(n):
                c += self.e_vals[x, i] * np.outer(cov_diff[x, :], np.transpose(cov_diff[x, :]))
            c = c / counts[i]
            self.cov[i] = c
            c = 0

    def update_cov2(self):
        counts = np.sum(self.e_vals, axis=0)
        reg_term = np.eye(self.data.shape[1]) * 1e-6
        for i in range(self.k):
            delta = self.data - self.means[i]
            Sj = np.dot(self.e_vals[:, i] * delta.T, delta) / counts[i]
            self.cov[i] = Sj + reg_term


    def log_likelihood(self):
        sum_ln = 0
        for k in range(self.k):
            w_term = np.log(self.weights[k])
            p_term = multivariate_normal.logpdf(self.data, self.means[k], self.cov[k], allow_singular=True)
            sum_ln += np.sum(self.e_vals[:, k] * (p_term + w_term))

        return sum_ln

    def e_step(self):
        n = self.data.shape[0]
        for j in range(self.k):
            self.e_vals[:, j] = np.dot(self.weights[j], multivariate_normal.pdf(self.data, self.means[j], self.cov[j], allow_singular=True))
        row_sums = self.e_vals.sum(axis=1)[:, np.newaxis]
        self.e_vals = self.e_vals / row_sums

    def m_step(self):
        n = self.data.shape[0]
        counts = np.sum(self.e_vals, axis=0)
        self.update_means()
        self.weights = counts / n
        self.update_cov()

    def m_step2(self):
        n = self.data.shape[0]
        counts = np.sum(self.e_vals, axis=0)
        self.update_means()
        self.weights = counts / n
        self.update_cov2()
