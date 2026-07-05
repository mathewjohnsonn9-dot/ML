import numpy as np

def top_k_sampling (logits, k):
    selected_index = np.argpartition(logits, -k)[-k:]
    selected_logits = logits[selected_index]
    max_ = np.max(selected_logits, axis=-1, keepdims= True)
    shifted = selected_logits-max_
    normalized_ = np.exp(shifted)/np.sum(np.exp(shifted), axis=-1, keepdims= True)
    return selected_index, normalized_
if __name__ == "__main__":
    logits = np.array([2.1, 0.3, 4.5, 1.2, 3.7])

    idx, probs = top_k_sampling(logits, k=3)

    print("Top-k indices:", idx)
    print("Top-k logits:", logits[idx])
    print("Probabilities:", probs)
    print("Sum of probabilities:", probs.sum())
