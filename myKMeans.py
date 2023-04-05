
import numpy as np

def kmeans(X, k, max_iters=100):
    n, d = X.shape
    centroids = X[np.random.choice(n, k, replace=False)]
    done = False
    i = 0
    while not done and (i < max_iters):
        new_centroids = centroids.copy()
        # assign each x to the closest centroid
        distances = np.sqrt(((X - centroids[:, np.newaxis]) ** 2).sum(axis=2))
        labels = np.argmin(distances, axis=0)

        # move each centroid to the mean of its assigned examples
        for j in range(k):
            new_centroids[j] = X[labels == j].mean(axis=0)

        if np.sum(np.square(centroids - new_centroids)) < 0.000001:
            done = True
        centroids = new_centroids
        i += 1
    # compute the total error
    total_error = np.sum((X - centroids[labels]) ** 2)
    return centroids, labels, total_error