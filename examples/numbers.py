from unet.config import Config
from unet.inference import Inference
import tensorflow as tf
import numpy as np
import visualize
import cv2

HEIGHT = 512
WIDTH = 512


class NumsConfig(Config):
    class_names = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Background"]


class NumClass:
    def __init__(self):
        self.config = NumsConfig()
        train_data, test_data = load_number_dataset(50)
        self.train_data = train_data
        self.test_data = test_data

    def run(self):
        x_train, y_train = self.train_data
        x_test, y_test = self.test_data

        inference = Inference(self.config)
        inference.create_model()
        inference.set_train_params(x_train, y_train)
        inference.train()
        inference.save_model('numbers.h5')

        inference.test_predictions(x_test, y_test)

    def visualize_dataset(self):
        x_train, y_train = self.train_data
        random_index = np.random.randint(0, len(x_train))

        input_example = x_train[random_index]
        mask_example = y_train[random_index]
        visualize.prepare_figure(
            input_example=input_example,
            mask_example=mask_example
        )


def load_number_dataset(length=100):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # x_train is an array containing images of 28x28 in grayscale
    # y_train in an array containing the actual number result
    # Reduce dataset to <length>
    x_train = x_train[:length]
    y_train = y_train[:length]
    x_test = x_test[:length]
    y_test = y_test[:length]

    train = transform_dataset(x_train, y_train)
    test = transform_dataset(x_test, y_test)
    return train, test


def transform_dataset(x, y):
    new_x_shape = (x.shape[0], HEIGHT, WIDTH, 3)
    new_y_shape = (y.shape[0], HEIGHT, WIDTH, 1)
    new_x = np.zeros(shape=new_x_shape)
    new_y = np.zeros(shape=new_y_shape)

    for i in range(len(x)):
        new_x[i] = create_three_channel_image(x[i])
        new_y[i] = create_mask_image(new_x[i], y[i])
    return new_x, new_y


def create_three_channel_image(np_array):
    w, h = np_array.shape
    img = np.zeros((w, h, 3))
    for i in range(w):
        for j in range(h):
            if np_array[(i, j)] <= 0:
                # set white background
                img[(i, j, 0)] = 255
                img[(i, j, 1)] = 255
                img[(i, j, 2)] = 255
    return cv2.resize(img, dsize=(HEIGHT, WIDTH))


def create_mask_image(np_array, class_value):
    w, h, _ = np_array.shape
    img = np.zeros((w, h))
    for i in range(w):
        for j in range(h):
            if np_array[(i, j, 0)] == 0:
                img[(i, j)] = class_value
            else:
                img[(i, j)] = 10  # background
    img = cv2.resize(img, dsize=(HEIGHT, WIDTH))
    return np.expand_dims(img, axis=-1)
