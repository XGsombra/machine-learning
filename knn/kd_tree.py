import numpy as np


def calculate_dist(x1, x2):
    diff = x1 - x2
    return np.sqrt(np.dot(diff.T, diff))


class KDTreeNode:
    """
    The class for kd-tree
    """

    def __init__(self, x):
        self.x = x
        self.axis_feature_idx = -1
        self.left = None
        self.right = None
        self.parent = None

    def __repr__(self):
        return KDTreeNode.print_kd_tree([self])

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

    @staticmethod
    def construct(data, feature_idx, parent=None):
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
        root.axis_feature_idx = feature_idx
        root.parent = parent
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
                np.array(left_nodes), next_feature_idx, root)
        if len(right_nodes) > 0:
            root.right = KDTreeNode.construct(
                np.array(right_nodes), next_feature_idx, root)
        return root

    def search(self, x):
        """
        Search to the nearest leaf node
        """
        curr = self
        # Search to the leaf node
        while curr.left is not None and curr.right is not None:
            if curr.left is None:
                curr = curr.right
            if curr.right is None:
                curr = curr.left
            curr = curr.left if x[curr.axis_feature_idx] < curr.x[curr.axis_feature_idx] else curr.right
        return curr

    @staticmethod
    def insert_sort(arr, curr_x, x):
        i = 0
        curr_x_x_dist = calculate_dist(x, curr_x)
        while i < len(arr) and calculate_dist(arr[i], x) < curr_x_x_dist:
            i += 1
        return arr[:i] + [curr_x] + arr[i:]

    def find_knn(self, x, k):
        """
        Given a data entry x who has the same size as the kd-tree nodes,
        return a list of the top k nearest neighbors of x
        """

        L = []
        searched_nodes = set([])
        curr = self
        while True:
            # Run a binary search
            curr = curr.search(x)
            # Add the node
            searched_nodes.add(curr)
            if len(L) < k + 1:
                L = KDTreeNode.insert_sort(L, curr.x, x)
                if len(L) == k + 1:
                    L.pop()
            # Go up
            while True:
                if curr.parent is None:
                    return L
                else:
                    prev = curr
                    curr = curr.parent
                    if curr not in searched_nodes:
                        searched_nodes.add(curr)
                        if len(L) < k + 1:
                            L = KDTreeNode.insert_sort(L, curr.x, x)
                            if len(L) == k + 1:
                                L.pop()
                        axis_feature_idx = curr.axis_feature_idx
                        dist_to_axis = abs(
                            curr.x[axis_feature_idx] - x[axis_feature_idx])
                        if dist_to_axis >= calculate_dist(L[-1], x) and len(L) == k:
                            continue
                        else:
                            curr = curr.left if prev == curr.right else curr.right
                            break


if __name__ == "__main__":
    data = np.random.rand(100, 5)
    tree = KDTreeNode.construct(data, 0)
    print(tree)

    k = 6
    target = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    expected = []
    for x in data:
        expected.append(calculate_dist(x, target))
    expected.sort()
    print(expected[:k])
    actual = []
    for x in tree.find_knn(target, k):
        actual.append(calculate_dist(x, target))
    print(actual)
