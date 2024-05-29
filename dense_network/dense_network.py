import matplotlib.pyplot as plt
import numpy as np
import argparse

def initialize_weights(input_size, hidden_size, output_size):
    W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / (input_size + hidden_size))
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / (hidden_size + output_size))
    b2 = np.zeros((1, output_size))
    return W1, b1, W2, b2


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.W1, self.b1, self.W2, self.b2 = initialize_weights(input_size, hidden_size, output_size)

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x <= 0, 0, 1)

    def forward(self, X):
        # Warstwa ukryta
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        # Warstwa wyjściowa
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.z2
        return self.a2

    def backward(self, X, y, output, learning_rate, clip_value=1):
        delta2 = output - y
        dW2 = np.dot(self.a1.T, delta2)
        db2 = np.sum(delta2, axis=0, keepdims=True)
        delta1 = np.dot(delta2, self.W2.T) * self.relu_derivative(self.z1)
        dW1 = np.dot(X.T, delta1)
        db1 = np.sum(delta1, axis=0)
        for dparam in [dW1, db1, dW2, db2]:
            np.clip(dparam, -clip_value, clip_value, out=dparam)
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    def train(self, X, y, epochs=1000, learning_rate=0.01, batch_size=32):
        num_samples = X.shape[0]
        for epoch in range(epochs):
            indices = np.random.permutation(num_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            for i in range(0, num_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                output = self.forward(X_batch)
                self.backward(X_batch, y_batch, output, learning_rate)
            loss = np.mean(np.square(y - self.predict(X)))
            if epoch % 1000 == 0:
                print(f'Epoch {epoch}, Additional Statistics:')
                print(f'Mean Squared Error (MSE): {loss}')
        print("Training completed.")

    def predict(self, X):
        return self.forward(X)


def load_data(filename):
    data = np.loadtxt(filename)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    X = data[:, :-1]
    y = data[:, -1].reshape(-1, 1)
    X_mean = np.mean(X)
    X_std = np.std(X)
    X_normalized = (X - X_mean) / X_std
    y_mean = np.mean(y)
    y_std = np.std(y)
    y_normalized = (y - y_mean) / y_std
    return X_normalized, y_normalized, y_mean, y_std


def split_data(X, y, train_ratio=0.8):
    num_train = int(train_ratio * len(X))
    X_train, y_train = X[:num_train], y[:num_train]
    X_test, y_test = X[num_train:], y[num_train:]
    return X_train, y_train, X_test, y_test


def start_network(filename, input_size, hidden_size, output_size, epochs, learning_rate):
    X, y, y_mean, y_std = load_data(filename)
    X_train, y_train, X_test, y_test = split_data(X, y)
    nn = NeuralNetwork(input_size, hidden_size, output_size)
    nn.train(X_train, y_train, epochs=epochs, learning_rate=learning_rate, batch_size=32)
    predictions = nn.predict(X_test)
    predictions_denormalized = predictions * y_std + y_mean
    y_test_denormalized = y_test * y_std + y_mean
    test_loss = np.mean(np.square(y_test_denormalized - predictions_denormalized))
    print(f'Mean Squared Error (MSE): {test_loss}')
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test_denormalized, color='blue', label='Rzeczywiste wartości')
    plt.scatter(X_test, predictions_denormalized, color='red', label='Przewidziane wartości')
    plt.title('Porównanie rzeczywistych i przewidywalnych wartości')
    plt.xlabel('Dane wejściowe')
    plt.ylabel('Dane wyjściowe')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a neural network on provided data.")
    parser.add_argument("filename", type=str, help="Path to the data file.")
    parser.add_argument("--input_size", type=int, default=1, help="Size of the input layer.")
    parser.add_argument("--hidden_size", type=int, default=10, help="Size of the hidden layer.")
    parser.add_argument("--output_size", type=int, default=1, help="Size of the output layer.")
    parser.add_argument("--epochs", type=int, default=50000, help="Number of epochs for training.")
    parser.add_argument("--learning_rate", type=float, default=0.05, help="Learning rate for training.")
    args = parser.parse_args()
    start_network(args.filename, args.input_size, args.hidden_size, args.output_size, args.epochs, args.learning_rate)
