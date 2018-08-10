import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm

INPUT = "image"
OUTPUT = "label"

CLASSES = {0: 'T-shirt/top',
           1: 'Trouser',
           2: 'Pullover',
           3: 'Dress',
           4: 'Coat',
           5: 'Sandal',
           6: 'Shirt',
           7: 'Sneaker',
           8: 'Bag',
           9: 'Ankle boot'
           }


def get_data(num_classes):
    data = pd.read_csv('fashion-mnist_train.csv')
    train_data_x = data.iloc[:, 1: 785]
    train_data_y = data.iloc[:, 0: 1]
    train_data_x = train_data_x / 255
    train_x = train_data_x.as_matrix()
    train_y = train_data_y.as_matrix()
    train_y = tf.contrib.keras.utils.to_categorical(train_y, num_classes)
    df = pd.DataFrame({INPUT: train_x, OUTPUT: train_y})
    return df


def batch(dset, batch_size=15):
    return [dset[i::batch_size] for i in range(batch_size)]


class FashionDataset:
    def __init__(self, dim, test_size=0.2):
        self.num_classes = 10
        dataset = get_data(self.num_classes)
        self.dim = dim
        self.flat = self.dim * self.dim
        train, test = train_test_split(dataset, test_size=test_size)
        self.train_x, self.train_y = [image for image in train[INPUT]], [
            label for label in train[OUTPUT]]
        self.test_x, self.test_y = [image for image in test[INPUT]], [
            label for label in test[OUTPUT]]
        assert len(self.train_x) == len(self.train_y)
        assert len(self.test_x) == len(self.test_y)

x = get_data(10)
print(x.head())