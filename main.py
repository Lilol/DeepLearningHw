import urllib.request
from enum import Enum
from os import remove, listdir
from os.path import join
from shutil import copytree
from numpy.random import seed, choice, permutation

import cv2
import patoolib
from matplotlib import pyplot as plt
from numpy import where, array, count_nonzero, asarray
from pyunpack import Archive
from tensorflow.keras.utils import to_categorical
from skimage.filters.rank import entropy
from skimage.morphology import disk

from scaler import Scaler


def download_and_unzip_images(target_directory):
    url = 'https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/vwdd9grvdp-2.zip'
    zip_file = join(target_directory, "images.zip")
    file_handle, _ = urllib.request.urlretrieve(url, zip_file)
    Archive(zip_file).extractall(target_directory)
    rar_file = join(target_directory, "Cloud-ImVN 1.0.rar")
    patoolib.extract_archive(rar_file, outdir=target_directory)
    copytree(join(target_directory, "Swimcat-extend"), target_directory)
    remove(zip_file)
    remove(rar_file)
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
    return array(X), array(Y)


def show_dataset(dataset, n_images_to_show_in_each_category=5):
    X, Y = dataset
    im = []
    for cat, lab in label.items():
        plt.figure()
        for i, c in enumerate(('b', 'g', 'r')):
            hist = cv2.calcHist(X[Y == lab], [i], None, [256], [0, 256])
            plt.plot(hist, color=c)
        plt.ylabel('No. of pixels')
        plt.xlabel('Intensity')
        plt.title(f"Global histogram of category: '{cat.value}'")
        plt.show()

        selected_images = choice(where(Y == lab)[0], n_images_to_show_in_each_category)
        concat = cv2.hconcat(X[selected_images])
        cv2.putText(concat, cat.value, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        im.append(concat)

    cv2.imshow("Five randomly selected images in each category", cv2.vconcat(im))
    cv2.waitKey(0)


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


def print_basic_infos(X):
    print(f"Number of images: {len(X)}")
    print(f"Size of images: {X[0].shape[0]}x{X[0].shape[1]}")
    print(f"Image type: {X[0].shape[2]}-channel, {X[0].dtype}")


def print_statistics(X, Y, name_of_dataset="dataset"):
    print(f"\nStatistics of {name_of_dataset}'")
    cat_len = 20
    stat_len = 24
    num_len = 10
    print(f"{'Category':{cat_len}}{'#images':>{num_len}}{'mean':>{stat_len}}{'std_dev':>{stat_len}}{'mean entropy':>{stat_len}}"
          f"{'entropy deviation':>{stat_len}}")
    entropies = {}
    entopy_error = []
    for cat, lab in label.items():
        images_of_category = X[Y == lab]
        global_entropy = entropy(cv2.cvtColor(cv2.hconcat(images_of_category), cv2.COLOR_BGR2GRAY), footprint=disk(10))
        mean_entropy = global_entropy.mean().mean()
        std_entropy = global_entropy.std()
        entropies[cat.value] = mean_entropy
        entopy_error.append(std_entropy)
        mean = images_of_category.mean()
        std = images_of_category.std()
        print(f"{cat.value:{cat_len}}{count_nonzero(Y==lab):{num_len}}{mean:{stat_len}.4f}{std:{stat_len}.4f}{mean_entropy:{stat_len}.4f}"
              f"{std_entropy:{stat_len}.4f}")

    return entropies, entopy_error


def plot_entropy(entropies, entropy_error):
    fig = plt.figure(figsize=(10, 5))
    # creating the bar plot
    plt.barh(list(entropies.keys()), list(entropies.values()), color='maroon', xerr=entropy_error)
    plt.ylabel("Categories")
    plt.xlabel("Global entropy")
    plt.title("Entropy of categories")
    plt.show()


def encode_labels(Y):
    return to_categorical(Y)


def randomize_input(X, Y):
    assert len(X) == len(Y)
    p = permutation(len(Y))
    return X[p], Y[p]


def main():
    seed(42)
    display_stats = True

    input_location = "E:\\work\\OneDrive_BME\\doktori\\3_felev\\Deep_learning\\nhf\\input"
    # Load the images and their respective categories from disk
    X, Y = load_data(input_location, download=False)

    if display_stats:
        # Show 5 random images from every category
        show_dataset((X, Y))
        # Display global data of the images
        print_basic_infos(X)
        # Display statistics of the data
        entropies, entropy_std = print_statistics(X, Y)
        # Plot the entorpy for the categories
        plot_entropy(entropies, entropy_std)

    # Randomize data
    X, Y = randomize_input(X, Y)
    # Transform labels to one-hot encoding
    Y = encode_labels(Y)
    # Split dataset into training, validation and test sets
    (X_train, Y_train), (X_valid, Y_valid), (X_test, Y_test) = split_dataset((X, Y), valid_split=0.2, test_split=0.1)
    # Rescale image channels
    X_train, X_valid, X_test = scale_dataset(X_train, X_valid, X_test)

    # Print statistics again, to check if mean and std-dev changed
    print_statistics(X_train, Y_train)


if __name__ == '__main__':
    main()
