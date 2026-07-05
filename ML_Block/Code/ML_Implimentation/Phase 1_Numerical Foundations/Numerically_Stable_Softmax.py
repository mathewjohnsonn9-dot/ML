import numpy as np


def softmax(x):
    if x.ndim == 1:
        out = x.reshape((1,-1))
    else:
        out =  x
    x_max = np.max(out, axis=1, keepdims=True)

    out = np.exp(out-x_max)/np.sum(np.exp(out-x_max), axis =1, keepdims=True)
    if x.ndim ==1:
        out = out.squeeze(0)
    return out


    # numerically stable
    # works for 1D and 2D (row-wise)
    # no for-loops
    return x
if __name__ =='__main__':
    x = np.array([[1000, 1001, 1002],
                  [1, 2, 3]])
    x = np.array([1000, 1001, 1002])


    x = softmax(x)
    print(x)