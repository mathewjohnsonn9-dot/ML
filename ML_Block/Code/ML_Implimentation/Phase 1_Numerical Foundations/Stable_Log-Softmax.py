import numpy as np

def log_softmax(x):
    if x.ndim == 1:
        x = x.reshape(1, -1)
    x_max = np.max(x, axis=1, keepdims= True)
    x_n = x-x_max
    x_ls = x_n - np.log(np.sum(np.exp(x_n), axis=1, keepdims=True))
    return x_ls
    """
    x: np.ndarray
       shape (n,) -> return (n,)
       shape (b,n) -> return (b,n) row-wise
    """
if __name__ =="__main__":
    x1 = np.array([1000, 1001, 1002])
    x2 = np.array([[1000, 1001, 1002],
                   [1, 2, 3]])

    print(log_softmax(x1))
    print(np.exp(log_softmax(x1)).sum())  # should be ~1

    out2 = log_softmax(x2)
    print(out2)
    print(np.exp(out2).sum(axis=1))  # should be ~[1,1]