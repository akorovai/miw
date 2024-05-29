import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def compare_models(data_file: str, test_size: float = 0.3, random_state: int = 1) -> None:
    a = np.loadtxt(data_file)
    x = a[:, [1]]
    y = a[:, [0]]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

    c_train = np.hstack([x_train ** 5, x_train ** 4, x_train ** 3, x_train ** 2, x_train, np.ones(x_train.shape)])
    v_train = np.linalg.inv(c_train.T @ c_train) @ c_train.T @ y_train

    c_test = np.hstack([x_test ** 5, x_test ** 4, x_test ** 3, x_test ** 2, x_test, np.ones(x_test.shape)])
    v_test = np.linalg.inv(c_test.T @ c_test) @ c_test.T @ y_test

    e_train = y_train - c_train @ v_train
    e_test = y_test - c_test @ v_test

    c_train_2 = np.hstack([x_train, np.ones(x_train.shape)])
    v_train_2 = np.linalg.pinv(c_train_2) @ y_train

    c_test_2 = np.hstack([x_test, np.ones(x_test.shape)])
    v_test_2 = np.linalg.pinv(c_test_2) @ y_test

    e_train_2 = y_train - c_train_2 @ v_train_2
    e_test_2 = y_test - c_test_2 @ v_test_2

    mse_train_model_1 = np.mean(e_train ** 2)
    mse_test_model_1 = np.mean(e_test ** 2)
    mse_train_model_2 = np.mean(e_train_2 ** 2)
    mse_test_model_2 = np.mean(e_test_2 ** 2)

    print("Porównanie modeli:")
    print("MSE Model 1 (train):", mse_train_model_1)
    print("MSE Model 1 (test):", mse_test_model_1)
    print("MSE Model 2 (train):", mse_train_model_2)
    print("MSE Model 2 (test):", mse_test_model_2)

    plt.figure(figsize=(10, 6))

    plt.scatter(x_train, y_train, color='blue', label='Dane treningowe')

    x_range_train = np.linspace(min(x_train), max(x_train), 100)
    y_model_1_train = sum(v_train[i] * x_range_train ** (5 - i) for i in range(6))
    plt.plot(x_range_train, y_model_1_train, color='red', label='Model 1 (wielomianowy) - trening')

    y_model_2_train = v_train_2[0] * x_range_train + v_train_2[1]
    plt.plot(x_range_train, y_model_2_train, color='green', label='Model 2 (liniowy) - trening')

    plt.scatter(x_test, y_test, color='orange', label='Dane testowe')

    x_range_test = np.linspace(min(x_test), max(x_test), 100)
    y_model_1_test = sum(v_test[i] * x_range_test ** (5 - i) for i in range(6))
    plt.plot(x_range_test, y_model_1_test, color='purple', linestyle='--', label='Model 1 (wielomianowy) - test')

    y_model_2_test = v_test_2[0] * x_range_test + v_test_2[1]
    plt.plot(x_range_test, y_model_2_test, color='pink', linestyle='--', label='Model 2 (liniowy) - test')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Porównanie Modeli')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    compare_models('../dane8.txt')
