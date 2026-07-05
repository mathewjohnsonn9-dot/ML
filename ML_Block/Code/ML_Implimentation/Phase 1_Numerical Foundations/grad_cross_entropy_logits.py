import numpy as np



def grad_cross_entropy_logits(logits, y_true):
    """
    logits: (b, C)
    y_true: (b,)
    returns: grad dL/dlogits with shape (b, C)
    where L is MEAN cross-entropy
    """
    r, c = logits.shape
    one_hot_code = np.zeros((r, c))
    one_hot_code[np.arange(r), y_true] = 1
    #one_hot_code.reshape(r, c)

    # computing the softmx:

    if logits.ndim  == 1:
        logits_out = logits.reshape(1, -1)
    else:
         logits_out = logits

    x_max =  np.max(logits_out, axis= 1, keepdims= True)
    logits_ =  logits_out -x_max
    softmax_ = np.exp(logits_)/np.sum(np.exp(logits_), axis=1, keepdims=True)
    gradient_ = (1/r) *(softmax_ - one_hot_code)

    return gradient_

if __name__ == "__main__":
    logits = np.array([[10, 0, -10],
                       [0, 10, -10]])
    y = np.array([0, 1])

    g = grad_cross_entropy_logits(logits, y)
    print(g)
    print("row sums:", g.sum(axis=1))