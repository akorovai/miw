import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from .plotka import plot_decision_regions


class Perceptron(object):
    def __init__(self, eta=0.01, n_iter=10):
        self.w_ = None
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])

        for _ in range(self.n_iter):
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)


def classify():
    iris = datasets.load_iris()

    x = iris.data[:, [2, 3]]
    y = iris.target

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1, stratify=y)

    perceptrons = []
    classes = np.unique(y)
    for cl in classes:
        if cl != 1:
            y_train_bin = np.where(y_train == cl, 1, -1)
            ppn = Perceptron(eta=0.1, n_iter=1000)
            ppn.fit(x_train, y_train_bin)
            perceptrons.append(ppn)

    plt.figure(figsize=(8, 6))
    for i, perc in enumerate(perceptrons):
        plot_decision_regions(X=x_train, y=y_train, classifier=perc)

    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.title("Perceptron Decision Boundaries ")
    plt.legend(classes, title="Iris Species")
    plt.show()


if __name__ == '__main__':
    classify()