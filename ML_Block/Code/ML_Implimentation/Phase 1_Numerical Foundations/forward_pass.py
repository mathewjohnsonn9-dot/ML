import  numpy as np
def linear_forward(X, W, b):
    Y = X @ W + b
    return Y
def linear_backward(dY, X, W):
    dx =  dY @ W.T
    dw = X.T @ dY
    db = np.sum(dY, axis=0)
    return dx, dw, db
    """
    dY: upstream gradient wrt Y, shape (b,k)
    returns: dX (b,d), dW (d,k), db (k,)
    """
if __name__ == '__main__':
    np.random.seed(0)
    X = np.random.randn(4, 3)
    W = np.random.randn(3, 2)
    b = np.random.randn(2)

    Y = linear_forward(X, W, b)
    dY = np.random.randn(4, 2)
    dX, dW, db = linear_backward(dY, X, W)

    print("Y shape:", Y.shape)
    print("dX shape:", dX.shape)
    print("dW shape:", dW.shape)
    print("db shape:", db.shape)
    print(Y)
