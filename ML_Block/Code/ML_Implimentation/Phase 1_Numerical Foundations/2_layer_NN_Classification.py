import numpy as np

class TwoLayerNNClassifier:
    def __init__(self, input_dim, hidden_dim, num_classes, lr=0.01, epochs=1000):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.lr = lr
        self.epochs = epochs

    def softmax(self, z3):
        max_ = np.max(z3, axis=1, keepdims= True)
        M= np.exp(z3 - max_)
        S = M/np.sum(M, axis= 1, keepdims= True)
        return  S
    def forward(self, X):
        z1 = X @ self.W1 +self.b1
        z2 = np.maximum(z1, 0)
        z3 = z2 @ self.W2 + self.b2
        y_hat = self.softmax(z3)
        cash = (z1, z2, z3)
        return y_hat, cash

        """
        X: (N, D)

        returns:
            logits: (N, C)
            cache: values needed for backward
        """
        pass

    def backward(self, X, y, cache):
        z1, z2, z3 = cache
        y_hat, _ = self.forward(X)
        N = X.shape[0]
        y_ = np.zeros(( N, self.num_classes))
        y_[np.arange(N), y] = 1
        y = y_
        dz3=  (1./N) * (y_hat - y)

        #  # softmax backward: S = softmax(P)
        #         dP = S * (dS - np.sum(dS * S, axis=-1, keepdims=True))  # (n,h,t,t)
        # dz3 = y_hat * (dy_hat- np.sum(dy_hat * y_hat, axis= -1, keepdims= True))
        dW2 = z2.T @ dz3
        db2 = np.sum(dz3, axis=0, keepdims= True)
        dz2 = dz3 @ self.W2.T
        dz1 = dz2 * (z1>0)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims= True)
        cash_ = (dW1, db1, dW2, db2)
        return cash_

        """
        X: (N, D)
        y: (N,) integer labels in {0, ..., C-1}
        cache: returned from forward()

        returns:
            dW1, db1, dW2, db2
        """


    def fit(self, X, y):
        self.W1 = np.zeros((self.input_dim, self.hidden_dim))
        self.b1 = np.zeros((1, self.hidden_dim))
        self.W2 = np.zeros((self.hidden_dim, self.num_classes))
        self.b2 = np.zeros((1, self.num_classes))
        for epoch in range(self.epochs):
            y_hat, cash = self.forward(X)
            cash_ = self.backward(X, y, cash)
            dW1, db1, dW2, db2 = cash_
            self.W1 = self.W1 - self.lr * dW1
            self.b1 = self.b1 - self.lr * db1
            self.W2 = self.W2 - self.lr * dW2
            self.b2 = self.b2 - self.lr * db2



        """
        X: (N, D)
        y: (N,)
        """


    def predict(self, X):
        y_hat, _ = self.forward(X)
        ans = np.argmax(y_hat, axis= 1)
        return  ans
        """
        X: (N, D)
        returns: (N,)
        """
