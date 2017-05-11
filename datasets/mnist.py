import os

from tensorflow.examples.tutorials.mnist import input_data

import reprod as rr

def load_mnist():
    expected_mnist_path = rr.paths.dropbox() + '/datasets/mnist/'
    if not os.path.isfile(expected_mnist_path):
        download_mnist(expected_mnist_path)
    raw_mnist = input_data.read_data_sets(expected_mnist_path, one_hot=True)
    import pdb;pdb.set_trace() 
    mnist_data = {'train': raw_mnist.train.images,
                  'val': raw_mnist.validation.images,
                  'test': raw_mnist.test.images}
    return mnist_data

def download_mnist(path):
    input_data.read_data_sets(path, one_hot=True)

     
