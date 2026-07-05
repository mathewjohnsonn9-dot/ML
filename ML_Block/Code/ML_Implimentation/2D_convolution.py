import numpy as np
def conv2d_forward(x, w, b, stride=1, padding=0):
    # x n,  chanel ,  high, width
    x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
    n,  c_in,  hc, wc =  x.shape
    c_outf, c_inf,  hf , wf = w.shape
    con_h = (hc  - hf) // stride + 1
    con_w = (wc  - wf) // stride + 1
    conv = np.zeros((n, c_outf, con_h, con_w))
    for i in range (n):
            for c in range(c_outf):
                for hc in range(con_h):
                    for wc in range (con_w):
                        hh = hc * stride; ww = wc * stride
                        patch = x[i, :, hh: hh+ hf, ww: ww+ wf]
                        conv[i, c, hc, wc] = np.sum(patch * w[c]) + b [c]
    return  conv
def backward_conv(R_out, x, w, b, stride=1, padding=0):
    n, d_out, d_h, d_w = R_out.shape
    x = np.pad(x, ((0,0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
    c_out, c_in, kh, kw = w.shape
    n, f_cin, f_h, f_w = x.shape
    x_d = np.zeros_like(x)
    w_d = np.zeros_like(w)
    db = np.zeros_like(b)

    for i in range(n):
        for c_ in range(d_out):
            for h_ in range(d_h):
                for w_ in range(d_w):
                    h_index = h_ * stride; w_index = w_ * stride
                    g = R_out[i, c_, h_, w_]; patch = x[i, :, h_index: h_index+kh, w_index: w_index+kw]
                    w_d[c_] += g * patch
                    x_d[i, :, h_index: h_index+kh, w_index: w_index+kw] += g * w[c_]
                    db[c_] +=g
    if padding>0:
        x_d = x_d[:, :, padding:-padding, padding:-padding]
    return db, w_d, x_d





if __name__ == "__main__":
    x = np.ones((1, 1, 4, 4))
    w = np.ones((1, 1, 3, 3))
    b = np.array([0.0])

    out = conv2d_forward(x, w, b, stride=1, padding=0)

    print(out.shape)   # expected: (1,1,2,2)
    print(out[0,0])    # expected: [[9,9],[9,9]]

    x = np.random.randn(2, 3, 5, 5)
    w = np.random.randn(4, 3, 3, 3)
    b = np.random.randn(4)

    out = conv2d_forward(x, w, b)

    R = np.random.randn(*out.shape)

    dw, dx, db = backward_conv(R, x, w, b)

    print(dx.shape)  # (2,3,5,5)
    print(dw.shape)  # (4,3,3,3)
    print(db.shape)  # (4,)