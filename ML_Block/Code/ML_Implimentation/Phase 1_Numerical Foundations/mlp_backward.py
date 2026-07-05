import numpy as np

def forward_pass(X, W1, b1, W2, b2):
    Z1 = X @ W1 + b1
    H = np.maximum(Z1, 0)
    Z2 = H @ W2 +b2
    cash = (W1, X,  b1, Z1, H, W2, b2)
    return Z2, cash

def mlp_backward(dZ2, cache):

    (W1, X,  b1, Z1, H, W2, b2)  = cache
    dW2 = H.T @ dZ2
    dH = dZ2 @ W2.T
    db2 = np.sum(dZ2, axis=0)
    # dZ1 = np.zeros_like(H)
    # dZ1[Z1>0] = 1
    dZ1 = dH * (Z1>0)
    dX = dZ1 @ W1.T
    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0)
    return dX , dW1 , db1 , dW2 , db2

    # Z1 = X @W1 +b1
    # H = Z1[Z1>0]


if __name__ == "__main__":
    np.random.seed(0)
    X = np.random.randn(4, 3)
    W1 = np.random.randn(3, 5)
    b1 = np.random.randn(5)
    W2 = np.random.randn(5, 2)
    b2 = np.random.randn(2)

    R = np.random.randn(4, 2)  # upstream for scalar loss
    parameter = [X, W1, b1, W2, b2]
    for i in range(len(parameter)):
        if parameter[i].ndim == 1:
            parameter[i] = parameter[i].reshape((1, -1))
    parameter_backup = parameter.copy()
    eps = 1e-5
    ans = [0]*len(parameter)
    for i  in range(len(parameter)):
        ans[i] = np.zeros_like(parameter[i])
        # parameter = parameter_backup.copy()
        # parameter_i = parameter_backup[i]
        shap = parameter[i].shape
        for k1 in range(shap[0]):
            for k2 in range(shap[1]):
                orgin = parameter[i][k1][k2]
                parameter[i][k1][k2]=orgin + eps
                L2, _ = forward_pass(*parameter)
                L2 = np.sum(L2 * R)
                parameter[i][k1][k2] = orgin
                parameter[i][k1][k2] = orgin - eps

                L1, _ = forward_pass(*parameter)
                L1 = np.sum(L1 * R)
                ans_ = (L2-L1)/(2*eps)
                ans[i][k1][k2] = ans_
                parameter[i][k1][k2] = orgin
    Z2, cache = forward_pass(X, W1, b1, W2, b2)
    W1, X, b1, Z1, H, W2, b2 = cache
    dX, dW1, db1, dW2, db2 = mlp_backward(R, cache)
    parameter = [X, W1, b1, W2, b2]
    max_err_W1 = np.max(np.abs(ans[1] - dW1))
    max_err_W2 = np.max(np.abs(ans[3]- dW2))
    max_err_b1 = np.max(np.abs(ans[2].squeeze(0) - db1))
    max_err_b2 = np.max(np.abs(ans[4].squeeze(0) - db2))

    print(max_err_W1)
    print(max_err_b1)
    print(max_err_W2)
    print(max_err_b2)

    # print("dX:", dX.shape)
    # print("dW1:", dW1.shape)
    # print("db1:", db1.shape)
    # print("dW2:", dW2.shape)
    # print("db2:", db2.shape)



    """
    dZ2: (b, k)  upstream gradient wrt logits (Z2)

    cache contains:
        X   : (b, d)
        Z1  : (b, h)
        H   : (b, h)
        W1  : (d, h)
        W2  : (h, k)

    returns:
        dX  : (b, d)
        dW1 : (d, h)
        db1 : (h,)
        dW2 : (h, k)
        db2 : (k,)
    """