import numpy as np
def mlp_forward(X, W1, b1, W2, b2):
    pre1 = X@W1+b1
    H = np.maximum(pre1,0)
    logits = H@W2+b2
    cache = (X, W1, b1, pre1, H, W2, b2)
    return logits, H
if __name__ == "__main__":
    np.random.seed(0)
    X = np.random.randn(5, 3)
    W1 = np.random.randn(3, 4)
    b1 = np.random.randn(4)
    W2 = np.random.randn(4, 2)
    b2 = np.random.randn(2)

    logits, cache = mlp_forward(X, W1, b1, W2, b2)
    print("logits shape:", logits.shape)
    print(logits[:2])
    print("cache items:", len(cache) if isinstance(cache, tuple) else cache.keys())
