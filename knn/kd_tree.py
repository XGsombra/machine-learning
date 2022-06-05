import numpy as np


class KDTreeNode:
    def __init__(self, x):
        self.x = x
        self.axis_idx = -1
        self.left = None
        self.right = None

    @staticmethod
    def print_kd_tree(stack):
        output = ""
        if len(stack) == 0:
            return output
        new_stack = []
        for kd_tree_node in stack:
            output += str(kd_tree_node.x) + " "
            if kd_tree_node.left is not None:
                new_stack.append(kd_tree_node.left)
            if kd_tree_node.right is not None:
                new_stack.append(kd_tree_node.right)
        output += "\n"
        return output + KDTreeNode.print_kd_tree(new_stack)

    def __repr__(self):
        return KDTreeNode.print_kd_tree([self])

    @staticmethod
    def construct(data, feature_idx):
        """ 
        DATA: A non-empty numpy array of data.
        FEATURE_IDX: The index at which the data are splited.

        Construct a kd-tree and return its root.
        """

        n = data.shape[0]
        k = data.shape[1]
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
        for i in range(n):
            if i == axis_idx:
                continue
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
    # data = np.random.rand(7, 2)
    # tree = KDTreeNode.construct(data, 0)
    # print(tree)
    pass
