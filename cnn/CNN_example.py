from tensorflow.keras.datasets import cifar10
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def train_and_evaluate_models():
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.7, random_state=42)

    animal_classes = [2, 3, 4, 5, 6, 7]
    vehicle_classes = [0, 1, 8, 9]

    train_labels_binary = to_categorical([1 if label in animal_classes else 0 for label in train_labels], num_classes=2)
    val_labels_binary = to_categorical([1 if label in animal_classes else 0 for label in val_labels], num_classes=2)
    test_labels_binary = to_categorical([1 if label in animal_classes else 0 for label in test_labels], num_classes=2)

    model_1 = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(2, activation='sigmoid')
    ])

    model_2 = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(2, activation='sigmoid')
    ])

    model_3 = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(2, activation='sigmoid')
    ])

    model_1.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    model_2.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    model_3.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    print("Training Model 1...")
    history_1 = model_1.fit(train_images, train_labels_binary, epochs=5, batch_size=64, validation_data=(val_images, val_labels_binary))

    print("Training Model 2...")
    history_2 = model_2.fit(train_images, train_labels_binary, epochs=5, batch_size=64, validation_data=(val_images, val_labels_binary))

    print("Training Model 3...")
    history_3 = model_3.fit(train_images, train_labels_binary, epochs=5, batch_size=64, validation_data=(val_images, val_labels_binary))

    val_acc_1 = history_1.history['val_accuracy'][-1]
    val_acc_2 = history_2.history['val_accuracy'][-1]
    val_acc_3 = history_3.history['val_accuracy'][-1]

    print("\nValidation accuracy:")
    print("Model 1:", val_acc_1)
    print("Model 2:", val_acc_2)
    print("Model 3:", val_acc_3)

    optimal_model = model_1 if val_acc_1 >= val_acc_2 and val_acc_1 >= val_acc_3 else model_2 if val_acc_2 >= val_acc_3 else model_3
    optimal_model_name = "Model 1" if val_acc_1 >= val_acc_2 and val_acc_1 >= val_acc_3 else "Model 2" if val_acc_2 >= val_acc_3 else "Model 3"
    print(f"\nOptimal model selected: {optimal_model_name}")

    print("\nEvaluating optimal model on test data...")
    test_loss, test_acc = optimal_model.evaluate(test_images, test_labels_binary)
    print('Test accuracy:', test_acc)
