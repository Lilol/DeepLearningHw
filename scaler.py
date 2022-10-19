import numpy as np
from numpy import array, asarray, zeros, zeros_like


class Scaler:
    def __init__(self, batch_size="all", per_channel=False):
        """
        :param batch_size: Number of images to standardize across
        :param per_channel: Compute standardization over number of images
        """
        self.batch_size = batch_size
        self.standardize_per_channel = per_channel
        self.mean = None
        self.std_dev = None

    def fit(self, images):
        if self.batch_size == "all":
            self.__calc_mean__std(images)

        batch_size = array(images).shape[0] if self.batch_size == "all" else self.batch_size
        output = zeros_like(images, dtype='float32')
        for k in range(0, array(images).shape[0], batch_size):
            output[k:k + batch_size, ...] = self.scale(images[k:k + batch_size, ...], mean=None, std_dev=None)

        return output

    def __calc_mean__std(self, images):
        if self.standardize_per_channel:
            self.mean = images.mean(axis=tuple(range(images.ndim - 1)))
            self.std_dev = images.std(axis=tuple(range(images.ndim - 1)))
        else:
            self.mean = images.mean()
            self.std_dev = images.std()

    def __get_mean(self, images, mean):
        if mean is None:
            if self.standardize_per_channel:
                return images.mean(axis=tuple(range(images.ndim-1)))
            else:
                return images.mean()
        elif mean == "stored":
            return self.mean
        else:
            return mean

    def __get_std_dev(self, images, std_dev):
        if std_dev is None:
            if self.standardize_per_channel:
                return images.std(axis=tuple(range(images.ndim-1)))
            else:
                return images.std()
        elif std_dev == "stored":
            return self.std_dev
        else:
            return std_dev

    def scale(self, images, mean="stored", std_dev="stored"):
        mean = self.__get_mean(images, mean)
        std_dev = self.__get_std_dev(images, std_dev)
        output = zeros_like(images, dtype='float32')
        if self.standardize_per_channel:
            for ch in range(images.shape[-1]):
                output[..., ch] = (images[..., ch] - mean[ch]) / std_dev[ch]
        else:
            output = (images - mean) / std_dev
        return output
