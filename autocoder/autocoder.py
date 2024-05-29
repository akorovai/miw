import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from keras import layers, models
from tensorflow import keras
from tensorflow.keras.datasets import mnist

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


def create_autoencoder(input_shape):
    encoder_input = layers.Input(shape=input_shape)
    x = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(encoder_input)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(8, (3, 3), padding='same', activation='relu')(x)
    encoded = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(4, (3, 3), padding='same', activation='relu')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2DTranspose(8, (3, 3), padding='same', activation='relu')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2DTranspose(1, (3, 3), padding='same', activation='sigmoid')(x)

    auto_encoder = models.Model(encoder_input, decoded)
    encoder = models.Model(encoder_input, encoded)

    auto_encoder.compile(optimizer='adam', loss='binary_crossentropy')
    auto_encoder.summary()
    return auto_encoder, encoder


def train_autoencoder(autoencoder, train_images, test_images, epochs, batch_size):
    return autoencoder.fit(train_images, train_images, epochs=epochs, batch_size=batch_size,
                           validation_data=(test_images, test_images))


def create_classifier(encoder):
    encoded_input = keras.Input(shape=(7, 7, 8))
    flatten = layers.Flatten()(encoded_input)
    dense = layers.Dense(128, activation='relu')(flatten)
    output = layers.Dense(10, activation='softmax')(dense)
    classifier = models.Model(encoded_input, output)
    classifier.summary()
    return classifier


def train_classifier(classifier, x_train_labels, y_train_labels, test_images, test_labels, epochs, batch_size, encoder):
    full_model = models.Model(encoder.input, classifier(encoder.output))
    full_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history_classifier = full_model.fit(x_train_labels, y_train_labels, epochs=epochs,
                                        batch_size=batch_size, shuffle=True,
                                        validation_data=(test_images, test_labels))
    return full_model, history_classifier


def evaluate_classifier(full_model, test_images, test_labels):
    score = full_model.evaluate(test_images, test_labels, verbose=0)
    return score


def start_encoder(autoencoder_epochs, autoencoder_batch_size, classifier_epochs, classifier_batch_size):
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
    test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

    autoencoder, encoder = create_autoencoder(input_shape=(28, 28, 1))
    history = train_autoencoder(autoencoder, train_images, test_images, autoencoder_epochs, autoencoder_batch_size)

    epochs = np.arange(len(history.history['loss'])) + 1
    plt.plot(epochs, history.history['loss'], label='train_loss')
    plt.plot(epochs, history.history['val_loss'], label='val_loss')
    plt.title('Loss during training of autoencoder')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    classifier = create_classifier(encoder)

    num_labels = int(0.1 * train_images.shape[0])
    x_train_labels = train_images[:num_labels]
    y_train_labels = train_labels[:num_labels]

    full_model, history_classifier = train_classifier(classifier, x_train_labels, y_train_labels, test_images, test_labels,
                                                      classifier_epochs, classifier_batch_size, encoder)

    plt.plot(history_classifier.history['loss'], label='train_loss')
    plt.plot(history_classifier.history['val_loss'], label='val_loss')
    plt.plot(history_classifier.history['accuracy'], label='train_accuracy')
    plt.plot(history_classifier.history['val_accuracy'], label='val_accuracy')
    plt.title('Loss and Accuracy during training of classifier')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

    score = evaluate_classifier(full_model, test_images, test_labels)
    classifier_loss = score[0]
    classifier_accuracy = score[1]

    print("Classifier:")
    print(f"  Test loss: {classifier_loss}")
    print(f"  Test accuracy: {classifier_accuracy}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train autoencoder and classifier on MNIST dataset.")
    parser.add_argument("--autoencoder_epochs", type=int, default=2, help="Number of epochs for autoencoder training.")
    parser.add_argument("--autoencoder_batch_size", type=int, default=32, help="Batch size for autoencoder training.")
    parser.add_argument("--classifier_epochs", type=int, default=25, help="Number of epochs for classifier training.")
    parser.add_argument("--classifier_batch_size", type=int, default=128, help="Batch size for classifier training.")

