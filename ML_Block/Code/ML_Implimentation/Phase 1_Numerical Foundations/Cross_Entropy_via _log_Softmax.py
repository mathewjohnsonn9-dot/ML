import numpy as np



def cross_entropy_from_logits_v2(logits, y_true):
    logits_max = np.max(logits, axis= 1, keepdims= True)
    logits_n = logits -logits_max
    softmax_  = - logits_n + np.log(np.sum(np.exp(logits_n), axis= 1, keepdims= True))
    batch =  logits.shape[0]
    cros_entropy = softmax_[np.arange(batch), y_true]
    cross_en  =  (1/batch) * np.sum(cros_entropy)
    return cross_en
if __name__ == "__main__":
    logits = np.array([[10, 0, -10],
                       [0, 10, -10]])
    y = np.array([0, 1])

    print(cross_entropy_from_logits_v2(logits, y))

    """
    Use log_softmax (stable) and gather the true class log-prob.
    Return mean loss.
    """