import numpy as np
def PCA (X , k):
    X = X-np.mean(X, axis = 0 , keepdims= True)
    COV_Matrix = X.T @ X / (X.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(COV_Matrix)
    index_ = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[index_]
    eigenvectors = eigenvectors [:, index_]
    projection = X @ eigenvectors[:, : k]
    return projection, eigenvalues, eigenvectors