import numpy as np
from Stump import Stump 

X = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]])
Y = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1])
W = np.ones(10) / 10

if __name__=="__main__":
    stump = Stump()
    stump.train(X, Y, W)
    print(stump)