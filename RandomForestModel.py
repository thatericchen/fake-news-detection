import numpy as np
from sklearn.tree import ExtraTreeClassifier
from sklearn.metrics import confusion_matrix


class RandomForestFakeNewsClassifier(object):
    def __init__(self, num_trees, max_depth, max_features):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.max_features = max_features

        self.decision_trees = [
            ExtraTreeClassifier(max_depth=self.max_depth, criterion="entropy")
            for i in range(self.num_trees)
        ]

    def bootstrapping(self, num_training, num_features):
        self.bootstraps_row_indices = []
        self.feature_indices = []
        self.out_of_bag = []

        for i in range(self.num_trees):
            total = set(list(range(num_training)))
            row_idx = np.random.choice(num_training, num_training, replace=True)
            num_feats_to_pick = int(self.max_features * num_features)
            col_idx = np.random.choice(num_features, num_feats_to_pick, replace=False)
            total = total - set(row_idx)
            self.bootstraps_row_indices.append(row_idx)
            self.feature_indices.append(col_idx)
            self.out_of_bag.append(total)

    def fit(self, X, y):
        self.bootstrapping(X.shape[0], X.shape[1])
        for i in range(self.num_trees):
            tree_rows = self.bootstraps_row_indices[i]
            tree_cols = self.feature_indices[i]
            x_sample = X[tree_rows][:, tree_cols]
            y_sample = y[tree_rows]
            self.decision_trees[i].fit(x_sample, y_sample)
        print("finished fitting--ready to predict")

    def predict(self, X):
        N = X.shape[0]
        predictions = np.zeros(N)
        votes = np.zeros((N, 2))

        for i in range(self.num_trees):
            tree = self.decision_trees[i]
            feature_subset = self.feature_indices[i]
            X_subset = X[:, feature_subset]
            tree_preds = tree.predict(X_subset)
            for j in range(N):
                votes[j, tree_preds[j]] += 1

        predictions = np.argmax(votes, axis=1)
        return predictions

    def predict_probs(self, X):
        N = X.shape[0]
        votes = np.zeros((N, 2))

        for n in range(self.num_trees):
            tree = self.decision_trees[n]
            feature = self.feature_indices[n]
            X_subset = X[:, feature]
            tree_pred = tree.predict(X_subset)
            for i in range(N):
                votes[i, tree_pred[i]] += 1

        probs = votes / self.num_trees
        return probs

    # METRICS FOR TREE PERFORMANCE
    def confusion_matrix(self, X, y_true):
        y_pred = self.predict(X)
        cm = confusion_matrix(y_true.flatten(), y_pred)

        TN = cm[0, 0]
        FP = cm[0, 1]
        FN = cm[1, 0]
        TP = cm[1, 1]

        precision = 0
        recall = 0
        f1 = 0

        if TP + FP > 0:  # 0 check
            precision = TP / (TP + FP)

        if TP + FN > 0:  # 0 check
            recall = TP / (TP + FN)

        if precision + recall > 0:  # 0 checl
            f1 = 2 * precision * recall / (precision + recall)

        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        return cm
