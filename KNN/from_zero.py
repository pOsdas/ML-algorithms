# Реализация с нуля

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class KNearestNeighbors:
    def __init__(self, neighbors=5, regression=False):
        self.y_train = None
        self.X_train = None
        self.neighbors = neighbors
        self.regression = regression

    def fit(self, X_train, y_train):
        self.X_train, self.y_train = X_train, y_train

    def _euclidean_distances(self, x_test_i):
        return np.linalg.norm(self.X_train - x_test_i, axis=1)

    def _make_predictions(self, x_test_i):
        distances = self._euclidean_distances(x_test_i)
        k_nearest_indexes = np.argsort(distances)[:self.neighbors]
        targets = self.y_train[k_nearest_indexes]

        return np.mean(targets) if self.regression else np.bincount(targets.astype(int)).argmax()

    def predict(self, X_test):
        return np.apply_along_axis(self._make_predictions, axis=1, arr=X_test)


iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNearestNeighbors(neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Точность модели:", accuracy)
