import numpy as np
import matplotlib.pyplot as plt 

class LVQCluster:
    
    @staticmethod
    def calculate_dist(x1, x2):
        diff = x1 - x2
        return np.sqrt(np.dot(diff, diff))
        
    @staticmethod
    def train(X, Y, k_label, epochs, step=0.1):
        """Return a numpy array of labels for x after clustered using LVQ algorithm.

        Args:
            X (Numpy array): input dataset
            Y (Numpy array): input labels
            k (int): number of clusters
        """
        k = len(k_label)
        N = X.shape[0]
        centers = X[np.random.choice(range(N), k, replace=False)]
        for i in range(epochs):
            x_idx = np.random.choice(N)
            x = X[x_idx]
            y = Y[x_idx]
            dists = []
            for j in range(k):
                dists.append(LVQCluster.calculate_dist(x, centers[j]))
            if y == k_label[np.argmin(dists)]:
                centers[j] += step * (x - centers[j])
            else:
                centers[j] -= step * (x - centers[j])
        return centers
                    
                
        
if __name__=="__main__":
    N = 40 # num of samples
    D = 2  # num of sample dimension
    Cs = 2 # num of sample class label
    k = 4  # num of cluster
    X = np.random.randint(1, 100, (N, D)) * 1.0
    Y = np.random.randint(0, Cs, N) * 1.0
    k_labels = np.random.randint(0, Cs, k)
    
    centers = LVQCluster.train(X, Y, k_labels, 1000)
    
    dists = np.ones((N, k))
    for i in range(N):
        for j in range(k):
            dists[i, j] = LVQCluster.calculate_dist(X[i], centers[j])
    labels = np.argmin(dists, axis=1)
    print(labels)
    
    C = ["r", "b", "g", "y"]
    type = [".", "v"]
    plt.figure()
    print([k,Cs])
    for i in range(k):
        for l in range(Cs):
            Xi = []
            for j in range(N):
                if labels[j] == i and Y[j] == l:
                    Xi.append(X[j])
            Xi = np.array(Xi)
            if len(Xi) > 0:
                plt.scatter(Xi[:,0], Xi[:,1], c=C[i], marker=type[l])
    plt.show()