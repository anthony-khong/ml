import pickle
import os

import numpy as np

import ml
from ml import fun

def cifar10_directory():
    return ml.paths.dropbox() + '/datasets/cifar-10-batches-py'

def load_cifar10():
    if not os.path.isfile(cifar10_directory() + '/data_batch_1'):
        download_cifar10()
    train_batches = map(load_batch, [1, 2, 3, 4, 5])
    stack_matrix = lambda x, y: np.vstack([x, y])
    return {'train': fun.reduce_values(stack_matrix, train_batches),
            'test': load_batch('test')}

def load_batch(batch_id):
    file_name = 'test_batch' if batch_id == 'test' else f'data_batch_{batch_id}'
    batch_path = cifar10_directory() + '/' + file_name
    with open(batch_path, 'rb') as f:
        batch = pickle.load(f, encoding='latin1')
    return {'features': batch['data'],
            'labels': ml.wrangle.one_hot(batch['labels'])}

def download_cifar10():
    cifar10_url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    download_path = os.path.expanduser('~/Downloads/cifar10.tar.gz')
    os.system(f'wget -c {cifar10_url} -O {download_path}')
    data_directory = ml.paths.dropbox() + '/datasets'
    os.system(f'tar -xvzf {download_path} -C {data_directory}')

if __name__ == '__main__':
    data = load_cifar10()

    for split, data_split in data.items():
        features, labels = data_split['features'], data_split['labels']
        print(f'Split: {split} | Feature: {features.shape} | Labels: {labels.shape}')
