import numpy as np
from numpy import array, asarray


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
        batch_size = array(images).shape[0] if self.batch_size == "all" else self.batch_size
        for k in range(0, array(images).shape[0], batch_size):
            if self.standardize_per_channel:
                for ch in range(images.shape[3]):
                    pixels = asarray(images[k:k+batch_size, :, :, ch]).astype('float32')
                    images[k:k + batch_size, :, :, ch] = (pixels - pixels.mean()) / pixels.std()
            else:
                pixels = asarray(images[k:k + batch_size, :, :, :]).astype('float32')
                images[k:k + batch_size] = (pixels - pixels.mean()) / pixels.std()

        if self.batch_size == "all":
            if self.standardize_per_channel:
                self.mean = np.zeros((images.shape[3],))
                self.std_dev = np.zeros((images.shape[3],))
                for ch in range(images.shape[3]):
                    pixels = asarray(images[:, :, :, ch]).astype('float32')
                    self.mean[ch] = pixels.mean()
                    self.std_dev[ch] = pixels.std()
            else:
                pixels = asarray(images).astype('float32')
                self.mean = pixels.mean()
                self.std_dev = pixels.std()
        return images

    def scale(self, images):
        if self.standardize_per_channel:
            for ch in range(images.shape[3]):
                pixels = asarray(images[:, :, :, ch]).astype('float32')
                images[:, :, :, ch] = (pixels - self.mean) / self.std_dev
        else:
            pixels = asarray(images).astype('float32')
            images = (pixels - self.mean) / self.std_dev
        return images
