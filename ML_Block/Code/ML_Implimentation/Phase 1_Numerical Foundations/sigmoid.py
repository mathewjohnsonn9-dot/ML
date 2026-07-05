import numpy as np
def sigmoid(x):
    ans  = np.empty_like(x)
    pos = x>=0
    neg = ~ pos
    ans[pos] = 1./(1. + np.exp(-x[pos]))
    xpx = np.exp(x[neg])
    ans[neg] = xpx/(xpx+1)
    return  ans

def BCE(logits,  y):

    ans = np.maximum(logits, 0) - logits * y + np.log(1+np.exp(-np.abs(logits)))
    ans = ans.mean()
    return ans
def Gradient_of_BCE(logits, y):
    b = logits.shape[0]
    G = (1/b) * (sigmoid(logits) - y)
    return G
if __name__ =="__main__":
    logits = np.array([1000.0, -1000.0, 0.0, 2.0, -2.0])
    y = np.array([1, 0, 1, 1, 0])
    ans = BCE(logits, y)
    ans = Gradient_of_BCE(logits, y)
    print(ans)
