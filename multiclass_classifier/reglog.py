import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from plotka import plot_decision_regions

class LogisticRegressionGD(object):
    def __init__(self, eta=0.01, n_iter=100000, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()

        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)

class SoftmaxClassifier:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.classifiers = []
        for _ in range(num_classes):
            self.classifiers.append(LogisticRegressionGD())

    def fit(self, X, y):
        for i in range(self.num_classes):
            y_binary = np.where(y == i, 1, 0)
            self.classifiers[i].fit(X, y_binary)

    def predict(self, X):
        class_scores = np.zeros((X.shape[0], self.num_classes))
        for i in range(self.num_classes):
            class_scores[:, i] = self.classifiers[i].net_input(X)

        # Normalize scores using the softmax function
        exp_scores = np.exp(class_scores - np.max(class_scores, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return np.argmax(probs, axis=1)


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_performance(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)


def main():
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    softmax_classifier = SoftmaxClassifier(num_classes=len(np.unique(y)))
    softmax_classifier.fit(X_train, y_train)

    y_pred_train = softmax_classifier.predict(X_train)
    y_pred_test = softmax_classifier.predict(X_test)

    print("Training Set Metrics:")
    evaluate_performance(y_train, y_pred_train)

    print("\nTest Set Metrics:")
    evaluate_performance(y_test, y_pred_test)

    plot_decision_regions(X_train, y_train, classifier=softmax_classifier)
    plt.xlabel('petal length [cm]')
    plt.ylabel('petal width [cm]')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    main()

