import urllib.request
from os import remove, listdir
from os.path import join
from shutil import copytree

import cv2
import patoolib
from numpy import array
from pyunpack import Archive

from definitions import Category, label


def download_and_unzip_images(target_directory):
    url = 'https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/vwdd9grvdp-2.zip'
    print(f"Downloading Swimcat database from '{url}'")
    zip_file = join(target_directory, "images.zip")
    file_handle, _ = urllib.request.urlretrieve(url, zip_file)
    Archive(zip_file).extractall(target_directory)
    rar_file = join(target_directory, "Cloud-ImVN 1.0.rar")
    patoolib.extract_archive(rar_file, outdir=target_directory)
    remove(zip_file)
    remove(rar_file)
    return join(target_directory, "Swimcat-extend")


def load_data(input_location, download=False):
    if download:
        input_location = download_and_unzip_images(input_location)
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
