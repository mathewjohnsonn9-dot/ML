
import numpy as np

def K_Means(X, K, iter= 0):
    index_ = np.random.choice(X.shape[0], size= K, replace = False)
    centroid = X[index_]
    while iter<1000:
        dis = np.linalg.norm(X[:, None, :]-centroid[None, :, :], axis=2)
        labels = np.argmin(dis, axis=1)
        new_centriod = np.zeros_like(centroid)
        for i in range(K):
           # new_centriod[i] = np.mean(X[labels==i], axis=0)
           if np.any(labels == i):
               new_centriod[i] = np.mean(X[labels == i], axis=0)
           else:
               new_centriod[i] = centroid[i]
        if np.allclose(new_centriod, centroid):
            break
        centroid = new_centriod
        iter +=1
    return  labels, centroid

if __name__ == "__main__":
    X = np.array([
        [1, 2],
        [1, 3],
        [2, 2],
        [8, 8],
        [9, 8],
        [8, 9]
    ], dtype=float)

    labels, centroids = K_Means(X, 2)
    print("labels:", labels)
    print("centroids:\n", centroids)





