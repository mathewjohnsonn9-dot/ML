import numpy as np

class DataLoader:

    def __init__(self, X, y, batch_size, shuffle=True):

        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.N = X.shape[0]

    def __iter__(self):

        if self.shuffle:
            self.indices = np.random.permutation(self.N)
        else:
            self.indices = np.arange(self.N)

        self.current = 0
        return self

    def __next__(self):

        if self.current >= self.N:
            raise StopIteration

        idx = self.indices[self.current:self.current + self.batch_size]

        X_batch = self.X[idx]
        y_batch = self.y[idx]

        self.current += self.batch_size

        return X_batch, y_batch