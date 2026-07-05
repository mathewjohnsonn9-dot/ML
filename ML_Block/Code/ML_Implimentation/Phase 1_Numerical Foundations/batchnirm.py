import numpy as np
def batchnorm_forward(x, gama, beta, eps = 1e-5):
    mu = np.mean(x, axis= 0, keepdims= True)
    var = np.mean((x-mu)**2, axis= 0,  keepdims= True)
    xhat =  (x- mu) * (1./(np.sqrt(var+eps)))
    inv = (1./(np.sqrt(var+eps)))
    y = gama * xhat + beta
    cash = (y, x, xhat, gama, beta, inv)
    return  cash , y
def batchnorm_backward(R, cash):
    y, x, x_hat, gama, beta, inv =  cash
    N = x.shape[0]
    dx_hat = R * gama
    d_gama = np.sum(R * x_hat, axis= 0)
    d_beta = np.sum(R, axis= 0)
    dx =  inv * (dx_hat - np.mean(dx_hat, axis= 0, keepdims= True) - x_hat * (np.mean(x_hat*dx_hat,  axis= 0, keepdims=True)))
    return dx, dx_hat, d_gama, d_beta
if __name__ == "__main__":
    pass







