import numpy as np

def softmax(x):

    x_  = x
    if x.ndim == 1:
        x_ = x.reshape(1, -1)
    x_max =  np.max(x_, axis= 1, keepdims= True)
    x_d = x_ - x_max
    softmax_ = np.exp(x_d)/np.sum(np.exp(x_d), axis= 1, keepdims=True)

    if x.ndim ==1:
        softmax_ = softmax_.squeeze(0)

    return  softmax_

def softmax_jvp(logits, v):
    if logits.ndim == 1:
        logits = logits.reshape(1, -1)
        v = v.reshape(1, -1)

    softmax_ = softmax(logits)
    b, c = logits.shape
    dot = np.sum(softmax_ *v, axis= 1, keepdims= True)
    ans = softmax_ * (v - dot)
    return ans



    """
    logits: (b, C)
    v:      (b, C)  (a vector for each sample)
    returns: (b, C)  J_softmax(logits) @ v  computed efficiently
    No loops.
    """

if __name__ == '__main__':
    logits = np.array([1., 2., 3.])
    v = np.ones_like(logits)

    print(softmax_jvp(logits, v))