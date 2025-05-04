import numpy as np


class MultinomialNBFakeNewsClassifier:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_log_priors_ = None
        self.feature_log_prob_ = None
        self.classes_ = None

    def fit(self, X, y):
        y = y.flatten()
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        # FIXED: Compute class counts correctly
        class_counts = np.array([np.sum(y == c) for c in self.classes_])
        self.class_log_priors_ = np.log(class_counts / np.sum(class_counts))

        # Compute feature log probabilities with smoothing
        self.feature_log_prob_ = np.zeros((n_classes, n_features))
        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            feature_count_c = X_c.sum(axis=0)
            total_count_c = feature_count_c.sum()
            smoothed_feature_count = feature_count_c + self.alpha
            smoothed_total = total_count_c + self.alpha * n_features
            self.feature_log_prob_[i] = np.log(smoothed_feature_count / smoothed_total)
        return self

    def predict_probs(self, X):
        log_probs = X @ self.feature_log_prob_.T
        log_probs += self.class_log_priors_
        max_log_probs = np.max(log_probs, axis=1, keepdims=True)
        log_probs -= max_log_probs
        probs = np.exp(log_probs)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs

    def predict(self, X):
        log_probs = X @ self.feature_log_prob_.T
        log_probs += self.class_log_priors_
        return self.classes_[np.argmax(log_probs, axis=1)]

    def evaluate(self, X, y, name):
        y = y.flatten()
        y_indices = np.searchsorted(self.classes_, y)
        y_proba = self.predict_probs(X)
        eps = 1e-15
        y_proba = np.clip(y_proba, eps, 1 - eps)
        N = y.shape[0]
        log_loss = -np.sum(np.log(y_proba[np.arange(N), y_indices])) / N
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)

        from sklearn.metrics import precision_score, recall_score, f1_score

        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)

        width = 70
        print("\n" + "=" * width)
        print(f"█ {name} Metrics ".ljust(width - 1, "█"))
        print("=" * width)
        print(f"║ Loss      : {log_loss:.4f}".ljust(width - 1) + "║")
        print(f"║ Accuracy  : {accuracy:.4f}".ljust(width - 1) + "║")
        print(f"║ Precision : {precision:.4f}".ljust(width - 1) + "║")
        print(f"║ Recall    : {recall:.4f}".ljust(width - 1) + "║")
        print(f"║ F1 Score  : {f1:.4f}".ljust(width - 1) + "║")
        print("=" * width + "\n")
