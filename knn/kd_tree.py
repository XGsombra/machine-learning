import numpy as np


class KDTreeNode:
    def __init__(self, x):
        self.x = x
        self.axis_idx = -1
        self.left = None
        self.right = None

    @staticmethod
    def construct(data, feature_idx):
        """ 
        DATA: A numpy array of data.
        FEATURE_IDX: The index at which the data are splited.

        Construct a kd-tree and return its root.
        """

        print(data)
        n = data.shape[0]
        k = data.shape[1]
        print([n, k])
        feature_column = data[:, feature_idx]
        axis_idx = np.argsort(feature_column)[n//2]
        axis = data[axis_idx]
        root = KDTreeNode(axis)
        root.axis_idx = axis_idx
        # Base case where data has only one entry.
        if n == 1:
            return root
        # Recursive case.
        left_nodes = []
        right_nodes = []
        np.delete(data, axis_idx)
        for i in range(n-1):
            if data[i, feature_idx] < axis[feature_idx]:
                left_nodes.append(data[i])
            else:
                right_nodes.append(data[i])
        next_feature_idx = 0 if feature_idx == k - 1 else feature_idx + 1
        if len(left_nodes) > 0:
            root.left = KDTreeNode.construct(
                np.array(left_nodes), next_feature_idx)
        if len(right_nodes) > 0:
            root.right = KDTreeNode.construct(
                np.array(right_nodes), next_feature_idx)
        return root


if __name__ == "__main__":
    data = np.random.rand(100, 1)
    tree = KDTreeNode.construct(data, 0)
    print("finish")
