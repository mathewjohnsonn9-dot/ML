import numpy as np



class LogisticRegression:
    def __init__(self, lr=0.1, epochs=1000):
        self.epochs = epochs
        self.lr = lr
    def fit(self, X, y):
        self.W = np.zeros((X.shape[1], 1))
        self.b = 0
        y = y.reshape(-1, 1)
        for epoch in range(self.epochs):
            p = self.forward(X)
            self.W = self.W - self.lr * (1. / X.shape[0]) * X.T @ (p - y)
            self.b = self.b - self.lr * (1. / X.shape[0]) * np.sum(p - y)


    def sigmoid(self, z):
            return 1./ (1+np.exp(-z))
    def forward (self, X):
        z = X @ self.W + self.b
        p = self.sigmoid(z)
        return p


    def predict_proba(self, X):
        z = X @ self.W + self.b
        return self.sigmoid(z)


    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs >= 0.5).astype(int)
