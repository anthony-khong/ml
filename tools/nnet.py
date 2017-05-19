import numpy as np
import tensorflow as tf

def make_weight(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def make_bias(n_biases):
    return tf.Variable(tf.constant(0.1, shape=[n_biases]))

def split_ixs_for_one_epoch(n_examples, minibatch_size):
    ixs = np.arange(n_examples)
    np.random.shuffle(ixs)
    n_minibatches = n_examples // minibatch_size
    for i in range(n_minibatches):
        start_ix, end_ix = minibatch_size*i, minibatch_size*(i+1)
        yield ixs[start_ix:end_ix]
