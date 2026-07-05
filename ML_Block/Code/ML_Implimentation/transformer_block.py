import  numpy as np
def transformer(x):
    def LN(x, gama, beta):
        eps = 1e-5
        mu =  np.mean(x, axis= -1, keepdims= True)
        inv_std = np.sqrt(1./(np.mean((x-mu)**2, axis= -1, keepdims= True)+eps))
        x_hat = (x - mu) * inv_std
        y =  gama * x_hat + beta
        return  x_hat ,  inv_std, y
    def LN_backward(dR, x_hat, gama, beta, inv_std):
        dd = dR.shape[-1]
        dx_hat = gama * dR
        dbeta =  np.sum(dR, axis=(0, 1))
        dgamma =  np.sum(dR * x_hat, axis= (0, 1))
        dx =  (inv_std) * (dx_hat - np.mean(dx_hat, axis= -1, keepdims= True) - x_hat * (np.mean(x_hat * dx_hat, axis= -1, keepdims= True)))
        return  dx, dx_hat, dgamma, dbeta
    def MHA_forward (x, wq, wk, wv, num_head, mask):
        Q = x  @ wq; K = x @ wk ; V = x @wv
        n,  t,  d = Q.shape
        d_head = d // num_head
        Q = Q.reshape(n,  t, num_head,  d_head ); Q = np.transpose(Q, (0, 2, 1, 3))
        K = K.reshape(n,  t, num_head,  d_head ); K = np.transpose(K, (0, 2, 1, 3))
        V = V.reshape(n, t, num_head, d_head); V= np.transpose(V, (0, 2, 1, 3))

        P = Q @ np.transpose(K, (0, 1, 3, 2)) *(1./np.sqrt(d_head))
        mask_casual = np.tril(np.ones((t, t),dtype=bool))[None, None, :, :]
        P = np.where(mask_casual, P, -1e10)
        mask = mask[:, None, None, :]
        P = np.where(mask, P, -1e10)
        max_ = np.max(P, axis=-1, keepdims=True)
        N = np.exp(P - max_)
        S = N/np.sum(N, axis= -1, keepdims= True)
        out =  S @ V
        out = np.transpose(out, (0, 2, 1, 3 ))
        # n, t, num_head, d_head =
        out =  np.reshape(out, (n, t, d))
        return out, S, P, Q, K, V
    def MHA_backward(R, x, wq, wk, wv, S, P, Q, K, V, num_head, mask):
        n, t, d = R.shape
        d_head = d // num_head

        # (n,t,d) -> (n,h,t,d_head)
        R = R.reshape(n, t, num_head, d_head).transpose(0, 2, 1, 3)
        # R = np.where(mask[:,: None], R, 0.0)

        # grads through Out = S @ V
        # V = (V.transpose(0, 2, 1, 3))
        # V = V.reshape(n, t, d)
        dS = R @ V.transpose(0, 1, 3, 2)  # (n,h,t,t)
        dV = S.transpose(0, 1, 3, 2) @ R  # (n,h,t,d_head)
        mask_val = mask[:, None, :, None]  # (n,1,t,1) -> broadcast to (n,h,t,d_head)
        dV = np.where(mask_val, dV, 0.0)

        # softmax backward: S = softmax(P)
        dP = S * (dS - np.sum(dS * S, axis=-1, keepdims=True))  # (n,h,t,t)

        mask_causal = np.tril(np.ones((t, t), dtype=bool))[None, None, :, :]  # (1,1,t,t)
        mask_pad = mask[:, None, None, :]  # (n,1,1,t)

        valid = mask_causal & mask_pad  # broadcast to (n,1,t,t)
        dP = np.where(valid, dP, 0.0)
        # IMPORTANT: exact mask gradient kill (because forward used np.where -> constant branch)
        # mask_causal = np.tril(np.ones((t, t), dtype=bool))[None, None, :, :]  # (1,1,t,t)
        # mask_pad = mask[:, None, None, :]  # (n,1,1,t)
        # dP = np.where(mask_causal, dP, 0.0)
        # dP = np.where(mask_pad, dP, 0.0)

        # P = (Q K^T)/sqrt(d_head)
        scale = 1.0 / np.sqrt(d_head)
        dQ = (dP @ K) * scale  # (n,h,t,d_head)
        dK = (dP.transpose(0, 1, 3, 2) @ Q) * scale  # (n,h,t,d_head)

        # merge heads: (n,h,t,d_head) -> (n,t,d)
        dQ = dQ.transpose(0, 2, 1, 3).reshape(n, t, d)
        dK = dK.transpose(0, 2, 1, 3).reshape(n, t, d)
        dV = dV.transpose(0, 2, 1, 3).reshape(n, t, d)

        # linear projections: Q=x@wq etc
        dwq = np.sum(x.transpose(0, 2, 1) @ dQ, axis=0)  # (d,d)
        dwk = np.sum(x.transpose(0, 2, 1) @ dK, axis=0)
        dwv = np.sum(x.transpose(0, 2, 1) @ dV, axis=0)

        dx = dQ @ wq.T + dK @ wk.T + dV @ wv.T  # (n,t,d)

        return dx, dwq, dwk, dwv
    def FF(x, w1, b1, w2, b2):
        H1= x @ w1 +b1
        H2 = np.maximum(0, H1)
        out = H2 @ w2 + b2
        return  H1, H2, out
    def FF_backward(R, x,  w1, b1, w2, b2, H1, H2):

        # dw2= np.sum(np.transpose(H2, (0, 2, 1)) @ R, axis=(0))
        dw2 = np.sum(np.transpose(H2, (0, 2, 1)) @ R, axis=0)  # (d_ff, d_model)
        db2 = np.sum(R, axis= (0,1))
        dH2 = R @ np.transpose(w2)
        dH1 = dH2 * (H1>0)
        # dw1 =   np.sum(np.transpose(x, (0, 2, 1)) @ dH1, axis = (0))
        dw1 = np.sum(np.transpose(x, (0, 2, 1)) @ dH1, axis=0)  # (d_model, d_ff)

        dx = dH1 @ np.transpose(w1)
        db1 = np.sum(dH1, axis= (0,1))


        return dx , dw1, db1, dw2, db2
    def transformer_forward(x, wq,wk,wv,num_head, mask, gama1, beta1, gama2, beta2, w1,b1, w2, b2):
         x_hat, inv_std1, z1 = LN(x, gama1, beta1)
         z2, S, P, Q, K, V = MHA_forward(z1,wq,wk,wv,num_head, mask)
         z3 = z2 + x
         z3_hat, inv_std2, z4 = LN(z3,gama2, beta2)
         H1, H2, z5 = FF(z4, w1,b1, w2, b2)
         z6 = z5 +z3
         return  z6, z5, z4, z3, z2, z1, H1, H2, S, P, Q, K, V, x_hat, z3_hat, inv_std1, inv_std2
    def transformer_backward(R, x, wq,wk,wv,num_head, mask, gama1, beta1,gama2, beta2, w1,b1, w2, b2):
        z6, z5, z4, z3, z2, z1, H1, H2, S, P, Q, K, V, x_hat, z3_hat , inv_std1, inv_std2 = transformer_forward(x, wq,wk,wv,num_head, mask, gama1, beta1, gama2, beta2, w1,b1, w2, b2)
        dz5 = R
        dz3 = R
        dz4 , dw1, db1, dw2, db2  = FF_backward(R, z4, w1, b1, w2, b2, H1, H2)
        print("dw1 shape", dw1.shape, "db1 shape", db1.shape, "dw2 shape", dw2.shape, "db2 shape", db2.shape)
        # dz4 += np.transpose(dz4) @ R
        # dx, dx_hat, dgamma, dbeta =  LN_backward(dR, x_hat, gama, beta):
        dz3_, dz3_hat, d_gamma2, d_beta2 = LN_backward(dz4, z3_hat, gama2, beta2, inv_std2)
        dz3 += dz3_
        dz2 = dz3
        dz1, dwq, dwk, dwv = MHA_backward(dz2, z1, wq, wk, wv, S, P, Q, K, V, num_head, mask)
        # dz1 = np.transpose(z1) @ dz2
        dx = dz3
        dx_, dx_hat, d_gamma1, d_beta1 = LN_backward(dz1, x_hat, gama1, beta1, inv_std1)
        dx += dx_
        # return  dx, dw1, db1, dw2, db2 , dwq, dwk, dwv
        return dx, dw1, db1, dw2, db2, dwq, dwk, dwv

    # d_gamma1, d_beta1, d_gamma2, d_beta2


    return transformer_forward, transformer_backward

import numpy as np

def scalar_loss(out, R):
    return float(np.sum(out * R))

def sample_indices(arr, num=30, seed=0):
    rng = np.random.default_rng(seed)
    return [tuple(rng.integers(0, s) for s in arr.shape) for _ in range(num)]

def finite_diff_param(transformer_forward, x, params, R, name, idxs, eps=1e-5):
    arr = params[name]
    g = np.zeros(len(idxs), dtype=float)
    for m, idx in enumerate(idxs):
        orig = arr[idx]
        arr[idx] = orig + eps
        out_p, *_ = transformer_forward(
            x,
            params["wq"], params["wk"], params["wv"], params["num_head"], params["mask"],
            params["gamma1"], params["beta1"], params["gamma2"], params["beta2"],
            params["w1"], params["b1"], params["w2"], params["b2"],
        )
        Lp = scalar_loss(out_p, R)

        arr[idx] = orig - eps
        out_m, *_ = transformer_forward(
            x,
            params["wq"], params["wk"], params["wv"], params["num_head"], params["mask"],
            params["gamma1"], params["beta1"], params["gamma2"], params["beta2"],
            params["w1"], params["b1"], params["w2"], params["b2"],
        )
        Lm = scalar_loss(out_m, R)

        arr[idx] = orig
        g[m] = (Lp - Lm) / (2 * eps)
    return g

if __name__ == "__main__":
    np.random.seed(0)

    b, t, d_model = 2, 4, 6
    num_head = 2
    d_ff = 4 * d_model

    # inputs
    x = np.random.randn(b, t, d_model)
    R = np.random.randn(b, t, d_model)

    # padding mask (mask keys)
    mask = np.array([
        [1, 1, 1, 0],
        [1, 1, 0, 0],
    ], dtype=bool)
    # mask = np.ones((b, t), dtype=bool)

    # params (small init helps gradcheck)
    params = {
        "num_head": num_head,
        "mask": mask,

        "wq": 0.01*np.random.randn(d_model, d_model),
        "wk": 0.01*np.random.randn(d_model, d_model),
        "wv": 0.01*np.random.randn(d_model, d_model),

        "gamma1": np.ones(d_model),
        "beta1": np.zeros(d_model),
        "gamma2": np.ones(d_model),
        "beta2": np.zeros(d_model),

        "w1": 0.01*np.random.randn(d_model, d_ff),
        "b1": 0.01*np.ones(d_ff),          # push ReLU away from 0
        "w2": 0.01*np.random.randn(d_ff, d_model),
        "b2": 0.01*np.random.randn(d_model),
    }

    forward, backward = transformer(None)

    # forward shape
    out, *_ = forward(
        x,
        params["wq"], params["wk"], params["wv"], params["num_head"], params["mask"],
        params["gamma1"], params["beta1"], params["gamma2"], params["beta2"],
        params["w1"], params["b1"], params["w2"], params["b2"],
    )
    print("out shape:", out.shape)  # (b,t,d_model)

    # analytic grads
    dx, dw1, db1, dw2, db2, dwq, dwk, dwv = backward(
        R, x,
        params["wq"], params["wk"], params["wv"], params["num_head"], params["mask"],
        params["gamma1"], params["beta1"], params["gamma2"], params["beta2"],
        params["w1"], params["b1"], params["w2"], params["b2"],
    )

    analytic = {"wq": dwq, "wk": dwk, "wv": dwv, "w1": dw1, "b1": db1, "w2": dw2, "b2": db2}

    # sampled finite-diff checks
    eps = 2e-5
    for name in ["wq", "wk", "wv", "w1", "w2", "b2"]:
        idxs = sample_indices(params[name], num=25, seed=1)
        gnum = finite_diff_param(forward, x, params, R, name, idxs, eps=eps)
        gana = np.array([analytic[name][idx] for idx in idxs], dtype=float)
        abs_err = np.max(np.abs(gnum - gana))
        rel_err = abs_err / (np.max(np.abs(gnum)) + 1e-12)
        print(f"{name:3s}  max_abs_err={abs_err:.3e}  max_rel_err={rel_err:.3e}")









# transformer_block_full_numpy.py
# Pure NumPy: Transformer block forward + backward + sampled numerical gradient check
#
# Shapes:
#   x:      (B, T, D)
#   mask:   (B, T)   True=real token, False=PAD
#   wq,wk,wv: (D, D)
#   w1:     (D, Dff)    Dff = 4*D (default)
#   b1:     (Dff,)
#   w2:     (Dff, D)
#   b2:     (D,)
#   gamma*, beta*: (D,)
#
# Block:
#   z1 = LN1(x)
#   z2 = MHA(z1)  (self-attn, causal + padding mask on keys)
#   z3 = z2 + x
#   z4 = LN2(z3)
#   z5 = FF(z4)
#   out = z5 + z3

# import numpy as np
#
#
# # -----------------------------
# # LayerNorm (per-token over last dim)
# # -----------------------------
# def ln_forward(x, gamma, beta, eps=1e-5):
#     # x: (B,T,D), gamma/beta: (D,)
#     mu = np.mean(x, axis=-1, keepdims=True)                      # (B,T,1)
#     var = np.mean((x - mu) ** 2, axis=-1, keepdims=True)         # (B,T,1)
#     inv_std = 1.0 / np.sqrt(var + eps)                           # (B,T,1)
#     x_hat = (x - mu) * inv_std                                   # (B,T,D)
#     y = x_hat * gamma[None, None, :] + beta[None, None, :]       # (B,T,D)
#     cache = (x_hat, gamma, inv_std)
#     return y, cache
#
#
# def ln_backward(dy, cache):
#     # dy: (B,T,D)
#     x_hat, gamma, inv_std = cache
#     dx_hat = dy * gamma[None, None, :]                           # (B,T,D)
#     dbeta = np.sum(dy, axis=(0, 1))                               # (D,)
#     dgamma = np.sum(dy * x_hat, axis=(0, 1))                      # (D,)
#
#     # Standard LN backward (per token)
#     # dx = inv_std * (dx_hat - mean(dx_hat) - x_hat*mean(dx_hat*x_hat))
#     mean1 = np.mean(dx_hat, axis=-1, keepdims=True)              # (B,T,1)
#     mean2 = np.mean(dx_hat * x_hat, axis=-1, keepdims=True)      # (B,T,1)
#     dx = inv_std * (dx_hat - mean1 - x_hat * mean2)              # (B,T,D)
#
#     return dx, dgamma, dbeta
#
#
# # -----------------------------
# # Multi-Head Self-Attention (no output projection here)
# # -----------------------------
# def mha_forward(x, wq, wk, wv, num_heads, mask):
#     # x: (B,T,D)  mask: (B,T) True=real token
#     B, T, D = x.shape
#     assert D % num_heads == 0
#     Dh = D // num_heads
#
#     # Linear projections
#     Q = x @ wq   # (B,T,D)
#     K = x @ wk
#     V = x @ wv
#
#     # Reshape to heads: (B,H,T,Dh)
#     Qh = Q.reshape(B, T, num_heads, Dh).transpose(0, 2, 1, 3)
#     Kh = K.reshape(B, T, num_heads, Dh).transpose(0, 2, 1, 3)
#     Vh = V.reshape(B, T, num_heads, Dh).transpose(0, 2, 1, 3)
#
#     # Logits: (B,H,T,T)
#     logits = (Qh @ Kh.transpose(0, 1, 3, 2)) * (1.0 / np.sqrt(Dh))
#
#     # Masks:
#     # causal mask: allow j <= i
#     causal = np.tril(np.ones((T, T), dtype=bool))[None, None, :, :]  # (1,1,T,T)
#     # padding mask on keys: mask_k[b, 1, 1, j]
#     mask_k = mask[:, None, None, :]                                  # (B,1,1,T)
#
#     # apply to logits
#     logits = np.where(causal, logits, -1e9)
#     logits = np.where(mask_k, logits, -1e9)
#
#     # stable softmax
#     m = np.max(logits, axis=-1, keepdims=True)
#     exp = np.exp(logits - m)
#     S = exp / np.sum(exp, axis=-1, keepdims=True)                    # (B,H,T,T)
#
#     # attention output per head
#     Oh = S @ Vh                                                      # (B,H,T,Dh)
#
#     # merge heads back: (B,T,D)
#     out = Oh.transpose(0, 2, 1, 3).reshape(B, T, D)
#
#     # (optional but recommended) zero out outputs for PAD query positions too
#     out = out * mask[:, :, None]
#
#     cache = (x, wq, wk, wv, Qh, Kh, Vh, S, causal, mask_k, num_heads, Dh, mask)
#     return out, cache
#
#
# def mha_backward(dout, cache):
#     # dout: (B,T,D)
#     (x, wq, wk, wv, Qh, Kh, Vh, S, causal, mask_k, num_heads, Dh, mask) = cache
#     B, T, D = x.shape
#
#     # zero out grad for PAD query positions (because we zeroed forward out)
#     dout = dout * mask[:, :, None]
#
#     # reshape dout to heads: (B,H,T,Dh)
#     dOh = dout.reshape(B, T, num_heads, Dh).transpose(0, 2, 1, 3)
#
#     # Oh = S @ Vh
#     dS = dOh @ Vh.transpose(0, 1, 3, 2)                              # (B,H,T,T)
#     dVh = S.transpose(0, 1, 3, 2) @ dOh                              # (B,H,T,Dh)
#
#     # softmax backward row-wise:
#     # dlogits = S * (dS - sum(dS*S))
#     tmp = np.sum(dS * S, axis=-1, keepdims=True)
#     dlogits = S * (dS - tmp)                                          # (B,H,T,T)
#
#     # IMPORTANT: zero gradients where logits were masked (causal/pad keys)
#     dlogits = np.where(causal, dlogits, 0.0)
#     dlogits = np.where(mask_k, dlogits, 0.0)
#
#     # logits = (Qh @ Kh^T) / sqrt(Dh)
#     dQh = (dlogits @ Kh) * (1.0 / np.sqrt(Dh))                        # (B,H,T,Dh)
#     dKh = (dlogits.transpose(0, 1, 3, 2) @ Qh) * (1.0 / np.sqrt(Dh))  # (B,H,T,Dh)
#
#     # merge head grads back to (B,T,D)
#     dQ = dQh.transpose(0, 2, 1, 3).reshape(B, T, D)
#     dK = dKh.transpose(0, 2, 1, 3).reshape(B, T, D)
#     dV = dVh.transpose(0, 2, 1, 3).reshape(B, T, D)
#
#     # Q = x @ wq etc.
#     X2 = x.reshape(-1, D)            # (B*T, D)
#     dQ2 = dQ.reshape(-1, D)
#     dK2 = dK.reshape(-1, D)
#     dV2 = dV.reshape(-1, D)
#
#     dwq = X2.T @ dQ2                 # (D,D)
#     dwk = X2.T @ dK2
#     dwv = X2.T @ dV2
#
#     dx = dQ @ wq.T + dK @ wk.T + dV @ wv.T                            # (B,T,D)
#     return dx, dwq, dwk, dwv
#
#
# # -----------------------------
# # FeedForward: x @ w1 -> relu -> @ w2
# # -----------------------------
# def ff_forward(x, w1, b1, w2, b2):
#     # x: (B,T,D), w1:(D,Dff), b1:(Dff,), w2:(Dff,D), b2:(D,)
#     H1 = x @ w1 + b1[None, None, :]          # (B,T,Dff)
#     H2 = np.maximum(0.0, H1)                 # (B,T,Dff)
#     out = H2 @ w2 + b2[None, None, :]        # (B,T,D)
#     cache = (x, w1, w2, H1, H2)
#     return out, cache
#
#
# def ff_backward(dout, cache):
#     x, w1, w2, H1, H2 = cache
#     B, T, D = x.shape
#     Dff = H2.shape[-1]
#
#     # flatten (B,T) to batch dimension
#     X2 = x.reshape(-1, D)            # (B*T, D)
#     dY = dout.reshape(-1, D)         # (B*T, D)
#     H2f = H2.reshape(-1, Dff)        # (B*T, Dff)
#     H1f = H1.reshape(-1, Dff)
#
#     dw2 = H2f.T @ dY                 # (Dff, D)
#     db2 = np.sum(dY, axis=0)         # (D,)
#
#     dH2 = dY @ w2.T                  # (B*T, Dff)
#     dH1 = dH2 * (H1f > 0.0)          # (B*T, Dff)
#
#     dw1 = X2.T @ dH1                 # (D, Dff)
#     db1 = np.sum(dH1, axis=0)        # (Dff,)
#
#     dx = (dH1 @ w1.T).reshape(B, T, D)
#     return dx, dw1, db1, dw2, db2
#
#
# # -----------------------------
# # Transformer Block
# # -----------------------------
# def transformer_forward(
#     x,
#     wq, wk, wv, num_heads, mask,
#     gamma1, beta1, gamma2, beta2,
#     w1, b1, w2, b2
# ):
#     # LN1
#     z1, ln1_cache = ln_forward(x, gamma1, beta1)
#
#     # MHA
#     z2, mha_cache = mha_forward(z1, wq, wk, wv, num_heads, mask)
#
#     # Residual
#     z3 = z2 + x
#
#     # LN2
#     z4, ln2_cache = ln_forward(z3, gamma2, beta2)
#
#     # FF
#     z5, ff_cache = ff_forward(z4, w1, b1, w2, b2)
#
#     # Residual
#     out = z5 + z3
#
#     cache = (x, z1, z2, z3, z4, z5, ln1_cache, mha_cache, ln2_cache, ff_cache, mask)
#     return out, cache
#
#
# def transformer_backward(
#     dout, cache,
#     wq, wk, wv,
#     gamma1, beta1, gamma2, beta2,
#     w1, b1, w2, b2
# ):
#     (x, z1, z2, z3, z4, z5, ln1_cache, mha_cache, ln2_cache, ff_cache, mask) = cache
#
#     # out = z5 + z3
#     dz5 = dout
#     dz3 = dout
#
#     # z5 = FF(z4)
#     dz4, dw1, db1, dw2, db2 = ff_backward(dz5, ff_cache)
#
#     # z4 = LN2(z3)
#     dz3_from_ln2, dgamma2, dbeta2 = ln_backward(dz4, ln2_cache)
#     dz3 = dz3 + dz3_from_ln2
#
#     # z3 = z2 + x
#     dz2 = dz3
#     dx_res1 = dz3
#
#     # z2 = MHA(z1)
#     dz1, dwq, dwk, dwv = mha_backward(dz2, mha_cache)
#
#     # z1 = LN1(x)
#     dx_ln1, dgamma1, dbeta1 = ln_backward(dz1, ln1_cache)
#
#     # total dx
#     dx = dx_res1 + dx_ln1
#
#     return (
#         dx,
#         dw1, db1, dw2, db2,
#         dwq, dwk, dwv,
#         dgamma1, dbeta1, dgamma2, dbeta2
#     )
#
#
# # -----------------------------
# # Sampled finite-difference gradcheck
# # -----------------------------
# def scalar_loss(out, R):
#     return np.sum(out * R)
#
#
# def sample_indices(shape, m=40, seed=0):
#     rng = np.random.default_rng(seed)
#     return [tuple(rng.integers(0, s) for s in shape) for _ in range(m)]
#
#
# def finite_diff_param(forward_scalar_fn, params, name, idxs, eps=1e-5):
#     arr = params[name]
#     g = np.zeros(len(idxs), dtype=float)
#     for i, idx in enumerate(idxs):
#         old = arr[idx]
#         arr[idx] = old + eps
#         L2 = forward_scalar_fn(params)
#         arr[idx] = old - eps
#         L1 = forward_scalar_fn(params)
#         arr[idx] = old
#         g[i] = (L2 - L1) / (2 * eps)
#     return g
#
#
# if __name__ == "__main__":
#     np.random.seed(0)
#
#     # small sizes for gradcheck
#     B, T, D = 2, 4, 6
#     H = 2
#     Dff = 4 * D
#     assert D % H == 0
#
#     x = np.random.randn(B, T, D)
#     R = np.random.randn(B, T, D)
#
#     # IMPORTANT: ensure every query row has at least one valid key after causal+pad
#     # easiest: make first token valid always, and keep some valid keys
#     mask = np.array([
#         [1, 1, 1, 0],
#         [1, 1, 1, 0],
#     ], dtype=bool)
#
#     # params
#     wq = 0.1 * np.random.randn(D, D)
#     wk = 0.1 * np.random.randn(D, D)
#     wv = 0.1 * np.random.randn(D, D)
#
#     w1 = 0.1 * np.random.randn(D, Dff)
#     b1 = np.zeros(Dff)
#     w2 = 0.1 * np.random.randn(Dff, D)
#     b2 = np.zeros(D)
#
#     gamma1 = np.ones(D)
#     beta1 = np.zeros(D)
#     gamma2 = np.ones(D)
#     beta2 = np.zeros(D)
#
#     # forward
#     out, cache = transformer_forward(
#         x,
#         wq, wk, wv, H, mask,
#         gamma1, beta1, gamma2, beta2,
#         w1, b1, w2, b2
#     )
#     print("out shape:", out.shape)
#
#     # backward analytic
#     grads = transformer_backward(
#         R, cache,
#         wq, wk, wv,
#         gamma1, beta1, gamma2, beta2,
#         w1, b1, w2, b2
#     )
#     (dx, dw1_a, db1_a, dw2_a, db2_a, dwq_a, dwk_a, dwv_a,
#      dgamma1_a, dbeta1_a, dgamma2_a, dbeta2_a) = grads
#
#     analytic = {
#         "wq": dwq_a, "wk": dwk_a, "wv": dwv_a,
#         "w1": dw1_a, "b1": db1_a, "w2": dw2_a, "b2": db2_a,
#         "gamma1": dgamma1_a, "beta1": dbeta1_a, "gamma2": dgamma2_a, "beta2": dbeta2_a
#     }
#
#     # gradcheck wrapper (scalar loss)
#     params = {
#         "x": x, "R": R, "mask": mask, "H": H,
#         "wq": wq, "wk": wk, "wv": wv,
#         "w1": w1, "b1": b1, "w2": w2, "b2": b2,
#         "gamma1": gamma1, "beta1": beta1, "gamma2": gamma2, "beta2": beta2
#     }
#
#     def forward_scalar(P):
#         out2, _ = transformer_forward(
#             P["x"],
#             P["wq"], P["wk"], P["wv"], P["H"], P["mask"],
#             P["gamma1"], P["beta1"], P["gamma2"], P["beta2"],
#             P["w1"], P["b1"], P["w2"], P["b2"]
#         )
#         return scalar_loss(out2, P["R"])
#
#     # sampled finite-diff checks
#     eps = 1e-5
#     for name in ["wq", "wk", "wv", "w1", "b1", "w2", "b2", "gamma1", "beta1", "gamma2", "beta2"]:
#         idxs = sample_indices(params[name].shape, m=40, seed=1)
#         gnum = finite_diff_param(forward_scalar, params, name, idxs, eps=eps)
#         gana = np.array([analytic[name][idx] for idx in idxs], dtype=float)
#
#         abs_err = np.max(np.abs(gnum - gana))
#         rel_err = abs_err / (np.max(np.abs(gnum)) + 1e-12)
#         print(f"{name:6s}  max_abs_err={abs_err:.3e}   max_rel_err={rel_err:.3e}")