import  numpy as np

def logsumexp(x):
    if  x.ndim == 1:
        x_out = x.reshape((1, -1))
    else:
        x_out = x
    x_max = np.max(x_out, axis=1, keepdims=True)
    x_sub =  np.log(np.sum(np.exp(x_out - x_max), axis=1, keepdims=True)) + x_max
    if x.ndim ==1:
        x_sub = x_sub.squeeze(0)
    return  x_sub


    """
    x: np.ndarray
       shape (n,) → return scalar
       shape (b,n) → return (b,) row-wise
    """
if __name__ == '__main__':
    x =  np.array([[1000, 1001, 1002],
              [1, 2, 3]])
    ans  =  logsumexp(x)
    print(ans)