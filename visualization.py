import cv2
from matplotlib import pyplot as plt
from numpy import where, count_nonzero
from numpy.random import choice
from skimage.filters.rank import entropy
from skimage.morphology import disk

from definitions import label


def show_dataset(dataset, n_images_to_show_in_each_category=5):
    X, Y = dataset
    im = []
    for cat, lab in label.items():
        selected_images = choice(where(Y == lab)[0], n_images_to_show_in_each_category)
        concat = cv2.hconcat(X[selected_images])
        cv2.putText(concat, cat.value, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        im.append(concat)

    cv2.imshow("Randomly selected images from each category", cv2.vconcat(im))
    cv2.waitKey(0)


def create_histogram_of_images(images, name):
    plt.figure()
    for i, c in enumerate(('b', 'g', 'r')):
        hist = cv2.calcHist(images, [i], None, [256], [0, 256])
        plt.plot(hist, color=c)
    plt.ylabel('No. of pixels')
    plt.xlabel('Intensity')
    plt.title(f"Global histogram of {name}")
    plt.show()


def print_histograms(X, Y):
    create_histogram_of_images(X, "Entire dataset")
    for cat, lab in label.items():
        create_histogram_of_images(X[Y == lab], cat.value)


def print_basic_infos(X):
    print(f"Number of images: {len(X)}")
    print(f"Size of images: {X[0].shape[0]}x{X[0].shape[1]}")
    print(f"Image type: {X[0].shape[2]}-channel, {X[0].dtype}")


def plot_statistics(X, Y, name_of_dataset="dataset"):
    print(f"\nStatistics of {name_of_dataset}")
    cat_len = 20
    stat_len = 24
    num_len = 10
    print(f"{'Category':{cat_len}}{'#images':>{num_len}}{'mean':>{stat_len}}{'std_dev':>{stat_len}}")
    m = X.mean()
    s = X.std()
    mean = {"Entire dataset": m}
    std_dev = [s]
    print(f"{'Entire dataset':{cat_len}}{X.shape[0]:{num_len}}{m:{stat_len}.4f}{s:{stat_len}.4f}")
    for cat, lab in label.items():
        images_of_category = X[Y == lab]
        m = images_of_category.mean()
        s = images_of_category.std()
        mean[cat.value] = m
        std_dev.append(s)
        print(f"{cat.value:{cat_len}}{count_nonzero(Y == lab):{num_len}}{m:{stat_len}.4f}{s:{stat_len}.4f}")

    plt.figure(figsize=(10, 5))
    # creating the bar plot
    plt.barh(list(mean.keys()), list(mean.values()), color='maroon', xerr=std_dev)
    plt.ylabel("Categories")
    plt.xlabel("Mean")
    plt.title("Statistics by categories")
    plt.show()


def plot_entropy(X, Y, name_of_dataset="dataset"):
    print(f"\nEntropy of {name_of_dataset}")
    cat_len = 20
    stat_len = 24
    print(f"{'Category':{cat_len}}{'mean entropy':>{stat_len}}{'entropy deviation':>{stat_len}}")
    global_entropy = entropy(cv2.cvtColor(cv2.hconcat(X), cv2.COLOR_BGR2GRAY), footprint=disk(10))
    mean_entropy = global_entropy.mean().mean()
    entropies = {"Entire dataset": mean_entropy}
    std_entropy = global_entropy.std()
    entropy_error = [std_entropy]
    print(f"{'Entire dataset':{cat_len}}{mean_entropy:{stat_len}.4f}{std_entropy:{stat_len}.4f}")
    for cat, lab in label.items():
        images_of_category = X[Y == lab]
        global_entropy = entropy(cv2.cvtColor(cv2.hconcat(images_of_category), cv2.COLOR_BGR2GRAY), selem=disk(10))
        mean_entropy = global_entropy.mean().mean()
        std_entropy = global_entropy.std()
        entropies[cat.value] = mean_entropy
        entropy_error.append(std_entropy)
        print(f"{cat.value:{cat_len}}{mean_entropy:{stat_len}.4f}{std_entropy:{stat_len}.4f}")

    plt.figure(figsize=(10, 5))
    # creating the bar plot
    plt.barh(list(entropies.keys()), list(entropies.values()), color='maroon', xerr=entropy_error)
    plt.ylabel("Categories")
    plt.xlabel("Global entropy")
    plt.title("Entropy of categories")
    plt.show()
