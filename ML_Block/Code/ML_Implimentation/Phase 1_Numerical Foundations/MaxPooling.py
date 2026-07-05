from collections import  defaultdict
import numpy as np
def MaxPooling_Forward(x, window, padding):
    x = np.pad(x,  ((0, 0), (0,0), (padding, padding), (padding, padding)), mode = 'constant')
    n, c_in, h, w = x.shape
    h_ = (h-window) // window +1
    w_ = (w-window) // window +1
    pooling_x =  np.zeros((n, c_in, h_, w_))
    argument_index = defaultdict(tuple)
    for i in range(n):
        for j in range (c_in):
            for k in range(h_):
                for l in range (w_):
                    c_x = l * window
                    r_x = k * window
                    pooling_x[i, j, k, l] = np.max(x[i, j, r_x:r_x+window, c_x:c_x+window])
                    flat_index = np.argmax(x[i, j, r_x:r_x+window, c_x:c_x+window])
                    local_h , local_w = np.unravel_index(flat_index, [window, window])
                    argument_index[(i, j, k, l)] = (r_x+local_h, c_x+ local_w)
    cash = (x, pooling_x, argument_index, window, padding)
    return pooling_x, cash

def MaxPooling_Backward(R, cash):
    (x, pooling_x, argument_index, window, padding) = cash
    dx = np.zeros_like(x)
    def upsample():
        Up_R = np.zeros_like(x)
        for item in argument_index:
            (n, c, h, w) = item
            (i, j) = argument_index[(n, c, h, w )]
            Up_R[n, c, i, j] += R[n, c, h, w]
        return  Up_R
    Up_R = upsample()
    dx = Up_R

    if padding > 0:
        dx = dx[:, :, padding:-padding, padding:-padding]
    else:
        dx = dx

    return dx


if __name__ == "__main__":
    x = np.array([[
        [[1, 3, 2, 1],
         [4, 6, 5, 2],
         [7, 2, 9, 3],
         [1, 8, 4, 0]]
    ]], dtype=float)

    out, cash = MaxPooling_Forward(x, window=2, padding=0)
    x_pad, pooling_x, arg_idx, window, padding = cash

    print(out.shape)            # (1,1,2,2)
    print(out[0, 0])            # [[6,5],[8,9]]
    print(arg_idx[(0, 0, 0, 0)])  # (1,1)
    print(arg_idx)