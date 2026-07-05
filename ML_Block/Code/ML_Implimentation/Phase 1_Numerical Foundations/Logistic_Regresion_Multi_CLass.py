import numpy as np

class SoftmaxRegression:

    def __init__(self, lr=0.1, epochs=1000):
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        c = np.max(y) + 1
        y_ = np.zeros((y.shape[0], c))
        y_[np.arange(y.shape[0]), y] =1
        y = y_
        self.W = np.zeros((X.shape[1], c))
        self.b = np.zeros((1,c))
        def forwrd(X):
            P = X @ self.W + self.b
            M = np.max(P, axis= 1, keepdims= True)
            N = np.exp(P-M)
            out = N/ np.sum(N, axis= 1, keepdims= True)
            return  out
        N = X.shape[0]
        for epoch in range(self.epochs):
            P = forwrd(X)
            dl = (1./ N) * (P-y)
            self.W = self.W - self.lr * (X.T @ dl)
            self.b = self.b - self.lr * np.sum(dl, axis = 0, keepdims=True)

    def predict_proba(self, X):
        P = X @ self.W + self.b
        M = np.max(P, axis=1, keepdims=True)
        N = np.exp(P - M)
        out = N / np.sum(N, axis=1, keepdims=True)
        return out

    def predict(self, X):
        out = self.predict_proba(X)
        ans = np.argmax (out, axis=1)
        return ans

