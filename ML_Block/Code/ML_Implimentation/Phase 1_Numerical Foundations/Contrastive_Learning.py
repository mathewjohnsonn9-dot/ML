import numpy as np

def info_nce_loss(z1, z2, temperature=0.1):
    z1 = z1/np.linalg.norm(z1, axis=1, keepdims=True)
    z2 = z2/ np.linalg.norm(z2, axis=1, keepdims=True)

    E = (z1 @ z2.T)/temperature
    max_ = np.max(E, axis= 1, keepdims= True)
    E = E-max_
    D = np.diag(E)
    N = -np.log (np.exp(D)/np.sum(np.exp(E), axis= 1, keepdims= True))
    L = np.mean(N)
    return L


    """
    z1: (N, D)
    z2: (N, D)

    returns:
        scalar loss
    """
