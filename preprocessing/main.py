from os import makedirs

import tensorflow as tf
from numpy.random import seed
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

from load_data import load_data
from preprocessing import scale_dataset, split_dataset, randomize_input
from visualization import show_dataset, print_basic_infos, plot_statistics, plot_entropy, print_histograms

seed(42)
display_stats = True
input_location = "C:\\input"


def main():
    print("Loading dataset...")
    makedirs(input_location, exist_ok=True)
    # Load the images and their respective categories from disk
    X, Y = load_data(input_location, download=True)

    if display_stats:
        # Display global data of the images
        print_basic_infos(X)
        # Show 5 random images from every category
        show_dataset((X, Y))
        # Visualize histogram of entire dataset and the categories each
        print_histograms(X, Y)
        # Display statistics of the data
        plot_statistics(X, Y)
        # Plot the entropy for the categories
        plot_entropy(X, Y)

    print("Preprocessing data...")
    # Randomize data
    X, Y = randomize_input(X, Y)
    # Split dataset into training, validation and test sets
    (X_train, Y_train), (X_valid, Y_valid), (X_test, Y_test) = split_dataset((X, Y), valid_split=0.2, test_split=0.1)
    # Rescale image channels
    X_train, X_valid, X_test = scale_dataset(X_train, X_valid, X_test)

    if display_stats:
        # Print statistics again, to check if mean and std-dev changed
        plot_statistics(X_train, Y_train, "training data")
        plot_statistics(X_valid, Y_valid, "validation data")
        plot_statistics(X_test, Y_test, "test data")

    print("Encoding and augmenting data...")
    # Transform labels to one-hot encoding
    Y_train = to_categorical(Y_train)
    Y_valid = to_categorical(Y_valid)
    Y_test = to_categorical(Y_test)

    # Data augmentation: add random flipping and rotation to the training set
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
    ])

    for i, X_train_im in enumerate(X_train[:, ...]):
        X_train[i] = data_augmentation(X_train_im, training=True)


if __name__ == '__main__':
    main()
