import numpy as np
def KNN(x, X, k, y):
     X_min = np.min(X, axis=0, keepdims= True)
     X_max = np.max (X, axis= 0, keepdims= True)
     X = (X -X_min) / (X_max -X_min)

     x = (x - X_min) / (X_max - X_min)
     # assuming that Labeld_X is already normalized otherwise we will treat them same as X
     distance  = np.linalg.norm(x-X , axis= 1)
     ans_index  = np.argsort(distance)
     ans_index = ans_index[:k]
     labels = y[ans_index]
     # Get unique values and their counts
     values, counts = np.unique(labels, return_counts=True)  #

     # Find the index of the maximum count
     max_count_index = np.argmax(counts)  #

     # Use the index to find the corresponding value
     ans_label = values[max_count_index]
     return ans_label
