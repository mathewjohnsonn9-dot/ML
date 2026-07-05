import  numpy as np
def AUC (prob):
    prob = prob[np.argsort(prob[:, 0])]

    pos_count = np.sum(prob[:, 1] == 1)
    N = prob.shape[0]
    neg_count = N - pos_count
    total = 0
    for i in range(N):
        if prob[i, 1] == 1:
            total += i+1
    auc = (total - ((pos_count)*(pos_count+1)/2))/(pos_count*neg_count)
    return auc
