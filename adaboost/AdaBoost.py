import numpy as np
from Stump import Stump 

X = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]])
Y = np.array([[1], [1], [1], [-1], [-1], [-1], [1], [1], [1], [-1]])

class BoostingTree:
    def __init__(self):
        self.W = None
        self.e = None
        self.classifier = None
        
    def train(self, X, Y, iteration=100):
        N = X.shape[0]
        self.W = np.ones((X.shape[0], 1)) / N
        F = []
        A = []
        for i in range(iteration):
            stump = Stump()
            e = stump.train(X, Y, self.W)
            if e >= .5:
                break
            a = np.log((1-e)/e)/2
            self.W = np.array([self.W[i] * np.exp(-a*stump.f(X[i])*Y[i]) for i in range(N)]) # can be optimized using vectorization
            self.W /= np.sum(self.W)
            print(self.W)
            F.append(stump.f)
            A.append(a)
            #print(stump)
        print(A)
        self.classifier = lambda x: 1 if sum([A[i] * F[i](x) for i in range(len(A))]) > 0 else -1
        return self

if __name__=="__main__":
    b = BoostingTree()
    b.train(X, Y, 3)
    print(b.classifier(X[3]))