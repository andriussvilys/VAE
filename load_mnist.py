import keras
from keras.datasets import mnist
import matplotlib.pyplot as plt

class MNISTDataLoader:
    def __init__(self):
        self.data = None
        self.labels = None
        self.load_data()

    def load_data(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        # Normalize image values from range [0:255] to [0:1]
        x_train = x_train / 255
        x_test = x_test / 255

        img_width = x_train.shape[1]
        img_height = x_train.shape[2]
        num_channels = x_train.shape[-1]
        x_train = x_train.reshape(x_train.shape[0], img_height, img_width, num_channels)
        x_test = x_test.reshape(x_test.shape[0], img_height, img_width, num_channels)
        self.data = (x_train, x_test)
        self.labels = (y_train, y_test)

    def plot_samples(self, num_samples=3):
        plt.figure(figsize=(10, 6))
        for i in range(num_samples):
            plt.subplot(1, num_samples, i + 1)
            plt.imshow(self.data[i][:, :, 0], cmap='gray')
            plt.axis('off')
        plt.show()
