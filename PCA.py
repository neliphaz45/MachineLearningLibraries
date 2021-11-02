"""
One of the greatest dimensionality reduction technique that helps to identify the correlations and patterns in the dataset so that it can be transformed into a dataset of significantly lower dimensions without any loss of information.

It is unsupervised statistical technique that is used to examine the interrelations among a set of variables.

With PCA, you are mapping the data in a higher dimensional space to data in lower dimension space, the variance or spread of the data in the lower space dimensional space should be maximum.


Steps:

1. Standardization of Data
2. Computing covariance matrix
3. Calculation of eigenvectors and eigenvalues
4. Computing the Principal components
5. Reducing the dimensions of the Data

"""

import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        #Mean
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # Covariance, function needs sample as columns
        cov = np.cov(X.T)

        # Eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # sort eigenvectors
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        # store first n components
        self.components = eigenvectors[0:self.n_components]

    def transform(self, X):
        X = X - self.mean

        return np.dot(X, self.components.T)