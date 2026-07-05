import numpy as np
def softmax_ce_forward(logits, y):
    max_ =  np.max(logits, axis=1, keepdims= True)
    n = np.exp(logits - max_)
    S = n/np.sum(n, axis= 1, keepdims= True)
    dimn  = S.shape
    logit = S[np.arange(dimn[0]), y]
    softmax_ce = -1* np.mean(np.log(logit))
    return S, softmax_ce
def backward_sftmax_ce(S, y):
    n = S.shape[0]
    one_hot = np.zeros_like(S)
    one_hot[np.arange(n), y] =1
    ans = (1/n)*(S-one_hot)
    return ans

if __name__ == "__main__":
    import numpy as np

    logits = np.array([
        [2.0, 1.0, 0.1],
        [0.5, 2.5, 0.3]
    ])  # shape (2, 3)

    y = np.array([0, 1])  # correct classes
    ans = softmax_ce_forward(logits, y)
    print(ans)

    # returns loss, cache