import numpy as np


def adam_step(params, grads, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    params, grads, m, v are dict[name -> np.ndarray]
    t is timestep (int)
    """
    for param in params:
        g = grads[param]
        m[param] = beta1 * m[param] + (1-beta1) * g
        v[param] = beta2 * v[param] + (1-beta2) * (g**2)
        # m_hat = m[param]/(1-beta1**t)
        # v_hat = v[param]/(1-beta2**t)
        m_hat = m[param]
        v_hat = v[param]
        params[param] = params[param] - lr * (m_hat/(np.sqrt(v_hat)+eps))
    return params, m, v

def quad_loss(w):
    return 0.5 * (w*w)

# initialize
w = np.array([10.0])
params = {"w": w.copy()}
m = {"w": np.zeros_like(w)}
v = {"w": np.zeros_like(w)}

lr = 0.1
for t in range(1, 301):
    grads = {"w": params["w"].copy()}  # grad = w
    params, m, v = adam_step(params, grads, m, v, t, lr=lr)

    if t in [1, 2, 5, 10, 50, 100, 300]:
        print(t, "w:", params["w"], "loss:", quad_loss(params["w"]))