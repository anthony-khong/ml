import pickle
import os
from functools import reduce

import numpy as np

import ml

def cifar10_directory():
    return ml.paths.dropbox() + '/datasets/cifar-10-batches-py'

def load_cifar10():
    if not os.path.isfile(cifar10_directory() + '/data_batch_1'):
        download_cifar10()
    def combine_batch(left, right):
        return {'features': np.vstack([left['features'], right['features']]),
                'labels': np.vstack([left['labels'], right['labels']])}
    return {'train': reduce(combine_batch, map(load_batch, [1, 2, 3, 4])),
            'val': load_batch(5),
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
    data =load_cifar10()

