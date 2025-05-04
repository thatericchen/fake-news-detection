import numpy as np


class FakeNewsClassifier(object):
    def __init__(self, learning_rate=0.05, epochs=1000, threshold=0.5):
        self.lr = learning_rate
        self.epochs = epochs
        self.threshold = threshold
        self.weights = []
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []

    def sigmoid(self, s):
        """
        Sigmoid with numerical stability
        """
        s_clip = np.clip(s, -20, 20)
        return 1 / (1 + np.exp(-s_clip))

    def bias_augment(self, X):
        """
        Returns X with a prepended column of 1's
        """
        return np.hstack((np.ones((X.shape[0], 1), dtype=float), X))

    def loss(self, y, h_x):
        """
        Returns binary cross entropy loss
        """
        eps = 1e-15
        h_x = np.clip(h_x, eps, 1 - eps)
        return np.sum((y * np.log(h_x)) + ((1 - y) * np.log(1 - h_x))) / -y.shape[0]

    def predict_probs(self, x_aug, theta):
        """
        Returns predicted probabilities for each point
        """
        return self.sigmoid(x_aug @ theta)

    def predict_labels(self, h_x, threshold):
        """
        Returns predicted label for each point
        """
        y_hat = np.zeros((h_x.shape[0], h_x.shape[1]))
        y_hat = np.where(h_x > threshold, 1, y_hat)
        return y_hat

    def gradient(self, x_aug, y, h_x):
        """
        Returns gradient of loss function w.r.t theta
        """
        return np.matmul(x_aug.T, (h_x - y)) / x_aug.shape[0]

    def accuracy(self, y, y_hat):
        """
        Returns the accuracy of the predicted labels y_hat
        """
        matches = np.zeros((y.shape[0], y.shape[1]))
        matches = np.where(y == y_hat, 1, matches)
        return np.sum(matches) / y.shape[0]

    def fit(self, X_train, y_train, X_val, y_val):
        """
        Train the model using gradient descent.
        """
        X_train_aug = self.bias_augment(X_train)
        self.weights = np.zeros((X_train_aug.shape[1], 1))

        for epoch in range(self.epochs):
            h_x = self.predict_probs(X_train_aug, self.weights)
            gradient = self.gradient(X_train_aug, y_train, h_x)
            self.weights -= self.lr * gradient

            if epoch % 100 == 0:
                self.update_evaluation_lists(X_train, y_train, X_val, y_val, epoch)

        return self.weights

    def update_evaluation_lists(self, x_train, y_train, x_val, y_val, epoch):
        """
        Append to self.train_loss, self.train_acc, self.val_loss, self.val_acc
        """
        train_loss, train_acc = self.evaluate(x_train, y_train, self.weights)
        val_loss, val_acc = self.evaluate(x_val, y_val, self.weights)
        self.train_loss.append(train_loss)
        self.train_acc.append(train_acc)
        self.val_loss.append(val_loss)
        self.val_acc.append(val_acc)
        if epoch % 100 == 0:
            print(
                f"""Epoch {epoch}:
	            train loss: {round(train_loss, 3)}	train acc: {round(train_acc, 3)}
	            val loss:   {round(val_loss, 3)}	val acc:   {round(val_acc, 3)}"""
            )

    def evaluate(self, x, y, theta):
        """
        Calculate and returns loss and accuracy
        """
        x_new = self.bias_augment(x)
        h_x = self.predict_probs(x_new, theta)
        y_hat = self.predict_labels(h_x, self.threshold)
        loss = self.loss(y, h_x)
        accuracy = self.accuracy(y, y_hat)
        return (loss, accuracy)
