from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Rescaling, Resizing
from tensorflow.keras import Input, Model
from tensorflow.keras.regularizers import L2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class ConvRGB:

    def __init__(self, input_shape, n_cut, rescaling = False):
        self.input_shape = input_shape
        self.n_cut = n_cut
        self.kernel_shape = [int(input_dim / self.n_cut) for input_dim in self.input_shape[:2]]
        self.model = self.set_colour_kernels(self.build_conv_model())

    def colours_as_pct(self, input_images):
        images = input_images.copy()
        images += 1 # avoid dividing by zero
        images = np.true_divide(images, images.sum(axis=3, keepdims=True))
        return images

    def build_conv_model(self):

        input = Input(shape = self.input_shape)
        if rescaling:

        output = Conv2D(3, kernel_size = self.kernel_shape, activation = 'relu', strides = self.kernel_shape, padding = 'same')(input)
        flattened_output = Flatten()(output)
        model = Model(input, flattened_output)

        return model

    def set_colour_kernels(self, model):

        w_ones = np.ones(self.kernel_shape)
        w_zeros = np.zeros(self.kernel_shape)

        red = np.stack([w_ones, w_zeros, w_zeros], axis = -1)
        green = np.stack([w_zeros, w_ones, w_zeros], axis = -1)
        blue = np.stack([w_zeros, w_zeros, w_ones], axis = -1)

        colour_weights = np.stack([red, green, blue], axis = 2)

        model.layers[1].set_weights([colour_weights, model.layers[1].get_weights()[1]])

        return model


    def summary(self):
        # plot weights
        fig, axes = plt.subplots(1, 3, figsize = (3, 1))
        red = self.model.layers[1].get_weights()[0][:, :, 0, :]
        blue = self.model.layers[1].get_weights()[0][:, :, 1, :]
        green = self.model.layers[1].get_weights()[0][:, :, 2, :]
        for i, ax, colour in zip(range(3), axes, [red, blue, green]):
          ax.imshow(colour)
          ax.set_title(f'W_{i}')
        plt.tight_layout()
        fig.suptitle('Weights')
        return self.model.summary()
