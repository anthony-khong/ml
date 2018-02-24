import os

import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

import ml
from ml import fun

def load_mnist():
    expected_mnist_path = ml.paths.dropbox() + '/datasets/mnist/'
    if not os.path.isfile(expected_mnist_path):
        download_mnist(expected_mnist_path)
    raw_mnist = input_data.read_data_sets(expected_mnist_path, one_hot=True)
    extract_tf_ds = lambda ds: {'features': ds.images, 'labels': ds.labels}
    train_batches = map(extract_tf_ds, [raw_mnist.train, raw_mnist.validation])
    stack_matrix = lambda x, y: np.vstack([x, y])
    return {'train': fun.reduce_values(stack_matrix, train_batches),
            'test': extract_tf_ds(raw_mnist.test)}

def download_mnist(path):
    input_data.read_data_sets(path, one_hot=True)

if __name__ == '__main__':
    data = load_mnist()

    for split, data_split in data.items():
        features, labels = data_split['features'], data_split['labels']
        print(f'Split: {split} | Feature: {features.shape} | Labels: {labels.shape}')
