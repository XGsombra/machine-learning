import numpy as np

class Stump:
    def __init__(self, step=1):
        self.step = step
        self.f = None
        self.threshold = None
        self.feature_idx = None
        self.high_one = None
        
    def __repr__(self):
        return "threshold is " + str(self.threshold) + "\n" +\
            "feature index is " + str(self.feature_idx) + "\n" +\
            "high one is " + str(self.high_one)
    
    @staticmethod
    def calculate_error_rate(X, Y, W, t, feature_idx, high_one=True):
        """
        Calculate the error rate for the threshold
        
        high_one is True when data higher than threshold is labeled as 1
        """
        e = 0
        for i in range(X.shape[0]):
            e += W[i] * 1 \
            if ((X[i, feature_idx] - t) * Y[i] < 0 if high_one else (X[i, feature_idx] - t) * Y[i] > 0) \
            else 0
        return e
     
    def train(self, X, Y, W, step=1):
        e = 1.0 # error rate
        for f_idx in range(X.shape[1]):
            low = min(X[:,f_idx])
            high = max(X[:,f_idx])
            t = low - step / 2 # to cover the seperation that groups all the data points
            while t <= high + step / 2:
                for high_one in [True, False]:
                    new_e = Stump.calculate_error_rate(X, Y, W, t, f_idx, high_one)    
                    if new_e < e:
                        e = new_e
                        self.threshold = t
                        self.feature_idx = f_idx
                        self.high_one = high_one
                t += step
        self.e = e
        self.f = lambda x: 1 if x > self.threshold else -1
        return self