import urllib.request
from enum import Enum
from os import remove, listdir
from os.path import join
from shutil import copytree
from numpy.random import seed, choice

import cv2
import patoolib
from numpy import where, array
from pyunpack import Archive
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def download_and_unzip_images(target_directory):
    url = 'https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/vwdd9grvdp-2.zip'
    zipfile = join(target_directory, "images.zip")
    filehandle, _ = urllib.request.urlretrieve(url, zipfile)
    Archive(zipfile).extractall(target_directory)
    rarfile = join(target_directory, "Cloud-ImVN 1.0.rar")
    patoolib.extract_archive(rarfile, outdir=target_directory)
    copytree(join(target_directory, "Swimcat-extend"), target_directory)
    remove(zipfile)
    remove(rarfile)
    remove(join(target_directory, "Swimcat-extend"))


class Category(Enum):
    CLEAR_SKY = "A-Clear Sky"
    PATTERNED_CLOUDS = "B-Patterned Clouds"
    THIN_WHITE_CLOUDS = "C-Thin White Clouds"
    THICK_WHITE_CLOUDS = "D-Thick White Clouds"
    THICK_DARK_CLOUDS = "E-Thick Dark Clouds"
    VEIL_CLOUDS = "F-Veil Clouds"


label = {
    Category.CLEAR_SKY: 1,
    Category.PATTERNED_CLOUDS: 2,
    Category.THIN_WHITE_CLOUDS: 3,
    Category.THICK_WHITE_CLOUDS: 4,
    Category.THICK_DARK_CLOUDS: 5,
    Category.VEIL_CLOUDS: 6,
}


def load_data(input_location, download=False):
    if download:
        download_and_unzip_images(input_location)
    X, Y = [], []
    for inner_dir in listdir(input_location):
        try:
            category = Category(inner_dir)
        except ValueError:
            continue
        for image in listdir(join(input_location, inner_dir)):
            X.append(cv2.imread(join(input_location, inner_dir, image)))
            Y.append(label[category])
    return X, array(Y)


def show_dataset(dataset, n_images_to_show_in_each_category=5):
    X, Y = dataset
    for cat, lab in label.items():
        selected_images = choice(where(Y == lab)[0], n_images_to_show_in_each_category)
        for idx_image in selected_images:
            cv2.imshow(X[idx_image], cat.value)


def scale_dataset(X_train, X_valid, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)

    scaler.fit_transform(X_train)
    scaler.fit_transform(X_valid)
    scaler.fit_transform(X_test)


def split_dataset(dataset, valid_split=0.2, test_split=0.1):
    X, Y = dataset
    n_images = len(X)
    X_train, Y_train = X[0:n_images * (1 - valid_split - test_split)], Y[0:n_images * (1 - valid_split - test_split)]
    X_valid, Y_valid = X[n_images * (1 - valid_split - test_split):n_images * (1 - test_split)], Y[n_images * (
                1 - valid_split - test_split):n_images * (1 - test_split)]
    X_test, Y_test = X[n_images * (1 - test_split):], Y[n_images * (1 - test_split):]
    return (X_train, Y_train), (X_valid, Y_valid), (X_test, Y_test)


def show_statistics(X, Y):
    pass


def one_hot_encode_labels(Y):
    return OneHotEncoder(drop='first').fit_transform(Y)


def main():
    seed(42)

    input_location = "E:\\work\\OneDrive_BME\\doktori\\3_felev\\Deep_learning\\nhf\\input"
    # Load the images and their respective categories from disk
    X, Y = load_data(input_location, download=False)

    # Show 5 random images from every category
    show_dataset((X, Y))
    # Display statistics of the data
    show_statistics(X, Y)

    # Transform labels to one-hot encoding
    Y = one_hot_encode_labels(Y)
    # Split dataset into training, validation and test sets
    training, validation, test = split_dataset((X, Y), valid_split=0.2, test_split=0.1)
    # Rescale image channels
    scale_dataset(training, validation, test)


if __name__ == '__main__':
    main()
