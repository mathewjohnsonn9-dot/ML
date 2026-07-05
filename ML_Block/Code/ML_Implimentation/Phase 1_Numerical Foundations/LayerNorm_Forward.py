import numpy as np
def layernorm_forward(X, gamma, beta, eps=1e-5):
    X_mean = np.mean(X, axis= 1, keepdims= True)
    variance_ = np.mean((X-X_mean)**2, axis= 1, keepdims= True)
    var = 1./np.sqrt(variance_ + eps)
    X_n = (X - X_mean)*var
    Y =  gamma * X_n + beta
    cache = {
        "xhat": X_n,
        "inv_std": var,
        "gamma": gamma
    }

    return Y, cache
    """
    X:     (b, d)
    gamma: (d,)
    beta:  (d,)
    returns:
        Y: (b, d)
        cache
    """
if __name__ == "__main__":
    np.random.seed(0)
    X = np.random.randn(3, 4)
    gamma = np.random.randn(4)
    beta = np.random.randn(4)

    Y, cache = layernorm_forward(X, gamma, beta)
    print("Y shape:", Y.shape)
    print("row means (of X_hat) ~0:", np.mean(cache["xhat"], axis=1))
    print("row vars  (of X_hat) ~1:", np.var(cache["xhat"], axis=1))
    print(Y)