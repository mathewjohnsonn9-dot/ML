import numpy as np

def layernorm_forward(X, gamma, beta, eps=1e-5):
    mu =  np.mean(X, axis= 1, keepdims= True)
    var_ = np.mean((X-mu)**2, axis= 1, keepdims= True)
    inv_std = 1.0/np.sqrt(var_+eps)
    Xhat = (X-mu)*inv_std
    Y =  Xhat * gamma + beta
    cash = (Xhat, gamma, beta, inv_std)
    return  Y, cash
def layernorm_backward(dY, chash):
     (Xhat, gamma, beta , inv_std) = chash
     dgamma = np.sum(dY * Xhat, axis= 0)
     dbeta =  np.sum(dY, axis= 0)
     dXhat =  dY * gamma
     dX = inv_std * (dXhat - np.mean(dXhat, axis= 1, keepdims=True) - Xhat * np.mean(dXhat * Xhat, axis=1, keepdims=True))
     return dX, dgamma, dbeta

def check_gradient (R, X, gamma, beta, eps=1e-5):
    parameters_list = [X, gamma, beta]
    derivative_list = []
    for parameters in parameters_list:
        if parameters.ndim == 1:
            parameters =parameters.reshape(1, -1)
        r, c =  parameters.shape
        deriv_parameter = parameters.copy()
        for i in range (r):
            for j in range(c):
                origin = parameters[i][j]
                parameters[i][j] = origin + eps
                Y2, cash2 = layernorm_forward(*parameters_list, eps=1e-5)
                L2 = np.sum(Y2 * R)
                parameters[i][j] = origin - eps
                Y1, cash1 = layernorm_forward(*parameters_list, eps=1e-5)
                L1 = np.sum(Y1 * R)
                deriv_parameter[i][j] = (L2 - L1)/(2*eps)
                parameters[i][j] = origin
        derivative_list.append(deriv_parameter)
    return  derivative_list


if __name__ == '__main__':
    np.random.seed(0)
    X = np.random.randn(3, 4)
    gamma = np.random.randn(4)
    beta = np.random.randn(4)
    R = np.random.randn(3, 4)
    dY = R
    Y, cache = layernorm_forward(X, gamma, beta)
    dXf, dgammaf, dbetaf = layernorm_backward(dY, cache)
    dX, dgamma, dbeta = check_gradient (R, X, gamma, beta, eps=1e-5)

    max_err_X = np.max(np.abs(dXf- dX))

    max_err_gamma = np.max(np.abs(dgammaf - dgamma))

    max_err_beta = np.max(np.abs(dbetaf - dbeta))

    print(max_err_X, max_err_gamma, max_err_beta)


