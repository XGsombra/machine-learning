import numpy as np
import matplotlib.pyplot as plt 

class KmeansCluster:
    
    @staticmethod
    def calculate_dist(x1, x2):
        diff = x1 - x2
        return np.sqrt(np.dot(diff, diff))
        
    @staticmethod
    def train(X, k):
        """Return a numpy array of labels for x after clustered using kmeans algorithm.

        Args:
            X (Numpy array): input dataset
            k (int): number of clusters
        """
        N = X.shape[0]
        centers = X[np.random.choice(range(N), k, replace=False)]
        labels = np.ones(N) * -1
        while True:
            dists = np.ones((N, k))
            for i in range(N):
                for j in range(k):
                    dists[i, j] = KmeansCluster.calculate_dist(X[i], centers[j])
            new_labels = np.argmin(dists, axis=1)
            if np.any(labels-new_labels):
                labels = new_labels
            else:
                return labels
                    
                
        
if __name__=="__main__":
    N = 40
    D = 2
    k = 4
    X = np.random.randint(1, 100, (N, D))
    labels = KmeansCluster.train(X, k)
    C = ["r", "b", "g", "y"]
    plt.figure()
    for i in range(k):
        Xi = []
        for j in range(N):
            if i == labels[j]:
                Xi.append(X[j])
        Xi = np.array(Xi)
        plt.scatter(Xi[:,0], Xi[:,1], c=C[i])
    plt.show()