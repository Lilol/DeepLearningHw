from numpy.random import permutation

from scaler import Scaler


def scale_dataset(X_train, X_valid, X_test):
    scaler = Scaler()
    X_train = scaler.fit(X_train)
    X_valid = scaler.scale(X_valid)
    X_test = scaler.scale(X_test)
    return X_train, X_valid, X_test


def split_dataset(dataset, valid_split=0.2, test_split=0.1):
    X, Y = dataset
    n_images = len(X)
    X_train, Y_train = X[0:int(n_images * (1 - valid_split - test_split))], Y[0:int(n_images * (1 - valid_split - test_split))]
    X_valid, Y_valid = X[int(n_images * (1 - valid_split - test_split)):int(n_images * (1 - test_split))], Y[int(n_images * (
                1 - valid_split - test_split)):int(n_images * (1 - test_split))]
    X_test, Y_test = X[int(n_images * (1 - test_split)):], Y[int(n_images * (1 - test_split)):]
    return (X_train, Y_train), (X_valid, Y_valid), (X_test, Y_test)


def randomize_input(X, Y):
    assert len(X) == len(Y)
    p = permutation(len(Y))
    return X[p], Y[p]
