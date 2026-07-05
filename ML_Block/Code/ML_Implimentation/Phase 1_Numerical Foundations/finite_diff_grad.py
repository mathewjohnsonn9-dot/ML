import numpy as np
def log_softmax(logits):
    if logits.ndim == 1:
        logits = logits.reshape(1, -1)
    logits_max =  np.max(logits, axis= 1, keepdims= True)
    logits_d =  logits - logits_max
    total_exp_sum =  np.sum(np.exp(logits_d), axis=1, keepdims= True)
    log_ans =  logits_d - np.log(total_exp_sum)
    return log_ans

def cross_entropy_from_logits(logits, y_true):
    log_softmax_logits =  log_softmax(logits)
    selected_comp = log_softmax_logits[np.arange(logits.shape[0]),  y_true]
    func = -np.mean(selected_comp)
    return func

def finite_diff_grad(x, y_true, eps=1e-5):
    r, c =  x.shape
    import numpy as np
    ans =  np.zeros_like(x)
    for i in range(r):
        for j in range(c):
            x2 = x.copy()
            x2[i, j] += eps
            x1 = x.copy()
            x1[i, j] -= eps
            f2 = cross_entropy_from_logits(x2, y_true)
            f1 = cross_entropy_from_logits(x1, y_true)
            ans[i, j] = (f2 - f1) / (2 * eps)

    return ans
    import numpy as np

def grad_cross_entropy_logits(logits, y_true):
    r, c = logits.shape

    # stable softmax
    m = np.max(logits, axis=1, keepdims=True)
    exps = np.exp(logits - m)
    probs = exps / np.sum(exps, axis=1, keepdims=True)

    # correct one-hot
    one_hot = np.zeros((r, c))
    one_hot[np.arange(r), y_true] = 1

    grad = (probs - one_hot) / r
    return grad

if __name__ == '__main__':
    np.random.seed(0)
    logits = np.random.randn(2, 3)
    y_true = np.array([0, 2])
    ana_grad = grad_cross_entropy_logits(logits, y_true)

    ans = finite_diff_grad(logits, y_true, eps=1e-5)

    err = np.abs(ana_grad - ana_grad)
    print("max error:", err.max())
    print("fd:\n", ana_grad)
    print("an:\n",  ans)
    print("row sums an:",  ans.sum(axis=1))




