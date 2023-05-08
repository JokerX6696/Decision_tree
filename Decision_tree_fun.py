class Node:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, label=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.label = label

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X, y):
        self.n_classes = len(set(y))
        self.n_features = X.shape[1]
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = [len(np.where(y == label)[0]) for label in range(self.n_classes)]
        # Stop conditions
        if depth == self.max_depth or n_samples < self.min_samples_split or len(set(y)) == 1:
            return Node(label=np.argmax(n_labels))

        # Select the best feature to split the data
        best_feature_idx, best_threshold = self._find_best_split(X, y, n_samples, n_features, n_labels)

        # Split the data and build subtrees recursively
        left_idxs = np.where(X[:, best_feature_idx] < best_threshold)[0]
        right_idxs = np.where(X[:, best_feature_idx] >= best_threshold)[0]

        left = self._build_tree(X[left_idxs], y[left_idxs], depth+1)
        right = self._build_tree(X[right_idxs], y[right_idxs], depth+1)

        return Node(feature_idx=best_feature_idx, threshold=best_threshold, left=left, right=right)

    def _find_best_split(self, X, y, n_samples, n_features, n_labels):
        best_feature_idx, best_threshold = None, None
        best_gini = 1

        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])

            for threshold in thresholds:
                # Split the data
                left_idxs = np.where(X[:, feature_idx] < threshold)[0]
                right_idxs = np.where(X[:, feature_idx] >= threshold)[0]

                # Calculate Gini impurity
                gini = self._calculate_gini(y, left_idxs, right_idxs, n_labels)

                # Update the best split if necessary
                if gini < best_gini:
                    best_feature_idx = feature_idx
                    best_threshold = threshold
                    best_gini = gini

        return best_feature_idx, best_threshold

    def _calculate_gini(self, y, left_idxs, right_idxs, n_labels):
        gini = 0

        for idxs in [left_idxs, right_idxs]:
            if len(idxs) == 0:
                continue
            labels = y[idxs]
            _, counts = np.unique(labels, return_counts=True)
            impurity = 1 - sum((count / len(labels))**2 for count in counts)
            gini += len(idxs) / len(y) * impurity

        return gini

    def predict(self, X):
        return np.array([self._predict_one(x) for x in X])

    def _predict_one(self, x):
        node = self.tree
        while node.label is None:
            if x[node.feature_idx] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.label
