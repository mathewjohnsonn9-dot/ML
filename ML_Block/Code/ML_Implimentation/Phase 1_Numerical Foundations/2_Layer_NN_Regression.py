import numpy as np

class TwoLayerNNRegressor:
    def __init__(self, input_dim, hidden_dim, lr=0.01, epochs=1000):
        self.input_dim = input_dim
        self.hidden_dim =  hidden_dim
        self.lr = lr
        self.epochs = epochs

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        z2 = np.maximum(z1, 0)
        y = z2 @ self.W2 + self.b2
        return y, z2, z1
        """
        X: (N, D)
        returns:
            y_hat: (N, 1)
            cache: values needed for backward
        """
        pass
    def backward(self, X, y):
        y_hat, z2, z1 = self.forward(X)
        N =  X.shape[0]
        dy = (2* 1./N) * (y_hat-y)
        dW2 = z2.T @ dy
        db2 = np.sum(dy, axis=0, keepdims=True)
        dz2 = dy @ self.W2.T
        p = np.where(z1>0, 1, 0)
        dz1 = dz2 * p
        dW1 =  (X.T @ dz1)
        db1 = (np.sum(dz1, axis= 0, keepdims=True))
        return dW1, db1, dW2, db2
        """
        X: (N, D)
        y: (N, 1)
        cache: from forward()

        returns gradients:
            dW1, db1, dW2, db2
        """
        pass
    def fit(self, X, y):
        self.W1 = np.zeros((self.input_dim, self.hidden_dim))
        self.b1 = np.zeros ((1, self.hidden_dim))
        self.W2 = np.zeros((self.hidden_dim, 1))
        self.b2 = np.zeros((1,1))
        for  epoch in range (self.epochs):
            dW1, db1, dW2, db2 = self.backward(X, y)
            self.W1 -= self.lr * dW1
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            self.b1 -= self.lr * db1

        """
        X: (N, D)
        y: (N, 1)
        trains with gradient descent
        """

    def predict(self, X):
        Y, _ , _ = self.forward(X)
        """
        X: (N, D)
        returns: (N, 1)
        """
        return Y