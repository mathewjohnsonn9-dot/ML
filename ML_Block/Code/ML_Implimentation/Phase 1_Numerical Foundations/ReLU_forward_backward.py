import numpy as np

def relu_forward(X):
  relu = np.maximum(X, 0)
  return relu, X

def relu_backward(dY, cache):
    dX = np.zeros_like(cache)
    dX [cache>0] = 1
    ans = dY * dX
    return  ans
    """
    dY: (b, d)
    cache: whatever you stored in forward
    returns: dX
    """
"""
X: (b, d)
returns: Y, cache
"""
if __name__ == '__main__':
    X = np.array([[-1.0, 2.0, 0.0],
                  [3.0, -4.0, 5.0]])

    Y, cache = relu_forward(X)

    dY = np.ones_like(X)
    dX = relu_backward(dY, cache)

    print("Y:\n", Y)
    print("dX:\n", dX)