"""

This is the implememtation of PCA one dimensionality reduction approach used in ML. 
Credit to Vikram

Source: https://www.linkedin.com/in/vikram--krishna/?miniProfileUrn=urn%3Ali%3Afs_miniProfile%3AACoAABp1ZbwBivKwyrpU14xyiJGzoIYVgaW8lWw
"""

from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy linalg import eigh
import  numpy as np

def kernelPCA(X, gamma, n_components):

    # X-> MxN dataset stored as an array where samples stored as rows and attributes as columns
    # gamma -> Afree parameter (coefficient) for the RBF kernel
    # n_components: the number of components to be returned


    # Calculating the squared Euclidian distances for every pair of points in the MxN dataset
    sq_dists = pdist(X, 'sqeuclidean')

    # converting the pairwise distance into a symmetric MxN matriix
    mat_sq_dists = squareform(sq_dists)

    # Computing the MxN kernel matrix
    K = exp(-gamma * mat_sq_dists)

    # Centering the symetric MxN kernel matrix
    N = K.shape[0]
    one_n = np.ones((N,N))/N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # Obtaining eigen values in descending order with corresponding eigenvectors from the symmetric matrix.
    eigvals, eigvecs = eigh(K)

    # obtaining the i eigenvectors corresponding to the i highest eigenvalues
    x_pc = np.column_stack((eigvecs[:,-i] for i in range(1,n_components + 1)))

    return x_pc