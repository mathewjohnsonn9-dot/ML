import  numpy as np
def forward_dropout(x, w, b,  p):
    y = w @ x  + b
    S = y.shape

    mask = np.random.binomial(1, 1-p, size= S)
    mask = (1./ (1-p)) * mask
    y = y * mask
    cash =  y, x, mask, w,  b
    return cash
def backward_dropout(R, cash):
    y, x, mask, w, b = cash
    mask_dr = R * mask
    dw = (mask_dr @ x.T)
    dx = (w.T @ mask_dr)
    db = np.sum(mask_dr, axis=1)
    return db, dw, dx

import numpy as np

# -----------------------------
# Your functions (unchanged)
# -----------------------------
def forward_dropout(x, w, b, p):
    y = w @ x + b
    S = y.shape
    mask = np.random.binomial(1, 1 - p, size=S)
    mask = (1.0 / (1 - p)) * mask
    y = y * mask
    cash = (y, x, mask, w, b)
    return cash

def backward_dropout(R, cash):
    y, x, mask, w, b = cash
    mask_dr = R * mask
    dw = (mask_dr @ x.T)
    dx = (w.T @ mask_dr)
    db = np.sum(mask_dr, axis=1, keepdims=True)   # <-- we'll check this (likely wrong axis)
    return db, dw, dx

# -----------------------------
# Deterministic forward (same math as yours, but reuse mask)
# -----------------------------
def forward_dropout_with_mask(x, w, b, mask):
    y = w @ x + b
    out = y * mask
    return out

def scalar_loss(out, R):
    return float(np.sum(out * R))

def sample_indices(shape, num=30, seed=0):
    rng = np.random.default_rng(seed)
    return [tuple(rng.integers(0, s) for s in shape) for _ in range(num)]

def finite_diff(arr, idx, forward_loss_fn, eps=1e-5):
    orig = arr[idx]
    arr[idx] = orig + eps
    Lp = forward_loss_fn()
    arr[idx] = orig - eps
    Lm = forward_loss_fn()
    arr[idx] = orig
    return (Lp - Lm) / (2 * eps)

if __name__ == "__main__":
    np.random.seed(0)

    # ---- choose shapes consistent with your forward: y = w @ x + b
    in_dim = 5
    out_dim = 4
    B = 6
    p = 0.2

    x = np.random.randn(in_dim, B)
    w = 0.1 * np.random.randn(out_dim, in_dim)

    # IMPORTANT: b should broadcast to (out_dim, B)
    # Either (out_dim, 1) or (out_dim,)
    b = 0.1 * np.random.randn(out_dim, 1)

    R = np.random.randn(out_dim, B)

    # ---- run YOUR forward ONCE to lock the randomness (mask)
    cash = forward_dropout(x, w, b, p)
    y, x_c, mask, w_c, b_c = cash

    # sanity shapes
    print("Shapes:")
    print("x:", x.shape, "w:", w.shape, "b:", b.shape, "mask:", mask.shape, "y:", y.shape)

    # ---- analytic grads from YOUR backward
    db_a, dw_a, dx_a = backward_dropout(R, cash)

    print("\nAnalytic grad shapes:")
    print("dx:", dx_a.shape, "dw:", dw_a.shape, "db:", np.shape(db_a))

    # ---- numeric grads using fixed mask
    def loss_current():
        out = forward_dropout_with_mask(x, w, b, mask)
        return scalar_loss(out, R)

    eps = 1e-5

    # check W
    idxs_w = sample_indices(w.shape, num=40, seed=1)
    err_w = 0.0
    for idx in idxs_w:
        gnum = finite_diff(w, idx, loss_current, eps)
        gana = dw_a[idx]
        err_w = max(err_w, abs(gnum - gana))
    print("\nW max_abs_err:", err_w)

    # check x
    idxs_x = sample_indices(x.shape, num=40, seed=2)
    err_x = 0.0
    for idx in idxs_x:
        gnum = finite_diff(x, idx, loss_current, eps)
        gana = dx_a[idx]
        err_x = max(err_x, abs(gnum - gana))
    print("x max_abs_err:", err_x)

    # check b
    idxs_b = sample_indices(b.shape, num=20, seed=3)
    err_b = 0.0
    for idx in idxs_b:
        gnum = finite_diff(b, idx, loss_current, eps)

        # db_a indexing depends on shape you return.
        # If db_a is (out_dim,1) you can use db_a[idx].
        # If db_a is (out_dim,) you should map idx (i,0) -> i.
        try:
            gana = db_a[idx]
        except Exception:
            gana = db_a[idx[0]]

        err_b = max(err_b, abs(gnum - gana))
    print("b max_abs_err:", err_b)

    # ---- quick diagnosis for db axis bug
    # For (out_dim,B), correct db is sum over batch axis=1 (keepdims=True to match b)
    db_correct = np.sum((R * mask), axis=1, keepdims=True)
    print("\nCompare your db to correct db (max abs diff):",
          np.max(np.abs(db_correct - db_a.reshape(db_correct.shape) if np.size(db_a)==np.size(db_correct) else 0.0)))