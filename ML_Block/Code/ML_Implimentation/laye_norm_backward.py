import  numpy as np

def layernorm_forward(X, gamma, beta):
    mu = np.mean(X, axis=1, keepdims= True)
    std_ = np.mean((X-mu)**2, axis=1, keepdims=True)
    epss = 1e-5
    inv_std = np.sqrt(1./(std_+epss))
    Xhat  =  inv_std * (X - mu)
    cash = Xhat, inv_std, gamma
    Y = gamma * Xhat + beta
    return  Y, cash

def layernorm_backward(dY, cache):
    xhat, inv_std, gamma = cache
    dgmma = np.sum(dY*xhat, axis= 0)
    dbeta  = np.sum(dY, axis= 0)
    dxhat = dY * gamma
    dX = inv_std * (dxhat - np.mean(dxhat, axis=1, keepdims=True) - xhat * np.mean((dxhat * xhat), axis=1, keepdims=True))
    return  dX, dgmma,  dbeta
    """
    dY: (b, d)
    cache: from forward with keys: xhat, inv_std, gamma
    returns:
        dX:     (b, d)
        dgamma: (d,)
        dbeta:  (d,)
    """
if __name__ == '__main__':
    np.random.seed(0)
    X = np.random.randn(3, 4)
    gamma = np.random.randn(4)
    beta = np.random.randn(4)
    R = np.random.randn(3, 4)
    dY = R

    Y, cache = layernorm_forward(X, gamma, beta)

    # dY = np.random.randn(3, 4)
    dX, dgamma, dbeta = layernorm_backward(dY, cache)

    print("dX :", dX)
    print("dgamma :", dgamma)
    print("dbeta :", dbeta)
