import numpy as np
def cross_entropy_from_logits(logits, y_true):
    x_max =  np.max(logits, axis= 1, keepdims= True)
    x_sub = logits - x_max
    x_total =  np.sum(np.exp(x_sub), axis= 1, keepdims=True)
    x_softmax = -logits + x_max + np.log(x_total)
    x_target = x_softmax[np.arange(y_true.shape[0]), y_true].reshape(-1, 1)
    x_mean = np.sum(x_target)/y_true.shape[0]
    return x_mean
    """
    logits: shape (b, C)
    y_true: int labels shape (b,) with values in [0, C-1]
    returns: scalar mean cross entropy
    """
if __name__ == '__main__':

    logits = np.array([[10, 0, -10],
                       [0, 10, -10]])
    y_true = np.array([0, 1])
    x_mean = cross_entropy_from_logits(logits, y_true)
    print(x_mean)
