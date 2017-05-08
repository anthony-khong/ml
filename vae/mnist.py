import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import distributions as dd

import reprod as rr

MNIST_PATH = rr.paths.dropbox() + '/datasets/mnist/'

def load_mnist(n_train=1000, n_test=1000):
    sample_ix = lambda n: np.random.choice(np.arange(n), size=n, replace=False)
    train_ix, test_ix = sample_ix(n_train), sample_ix(n_test)
    mnist = input_data.read_data_sets(MNIST_PATH, one_hot=True)
    data = {'train': mnist.train.images[train_ix],
            'test': mnist.test.images[test_ix]}
    return data

def plot(axis, pixels):
    pixels = np.array(255 * pixels, dtype='uint8').reshape((28, 28))
    axis.imshow(pixels, cmap='gray')

if __name__ == '__main__':
    data = load_mnist()

    n_feats, n_hiddens, n_latents = 784, 100, 10
    make_weight = lambda shape: tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    make_bias = lambda n: tf.Variable(tf.constant(0.1, shape=[n]))

    # Encoder
    feats = tf.placeholder(tf.float32, [None, 784])
    weights = {'hidden': make_weight((n_feats, n_hiddens)),
               'mean': make_weight((n_hiddens, n_latents)),
               'std': make_weight((n_hiddens, n_latents))}
    biases = {'hidden': make_bias((n_hiddens)),
              'mean': make_bias((n_latents)),
              'std': make_bias((n_latents))}

    hidden = tf.matmul(feats, weights['hidden']) + biases['hidden']
    mean = tf.matmul(hidden, weights['mean']) + biases['mean']
    std = tf.exp(tf.matmul(hidden, weights['std']) + biases['std'])

    noise = tf.random_normal([tf.shape(feats)[0], n_latents])
    latents = mean + noise*std

    # Decoder
    weights_ = [make_weight((n_latents, n_hiddens)),
                make_weight((n_hiddens, n_feats))]
    biases_ = [make_bias((n_hiddens)), make_bias((n_feats))]

    hidden_ = tf.nn.relu6(tf.matmul(latents, weights_[0]) + biases_[0])
    preds = tf.sigmoid(tf.matmul(hidden_, weights_[1]) + biases_[1])

    # Objective function
    log_likelihood = feats*tf.log(preds) + (1.0-feats)*tf.log(1.0-preds)
    prior_divergence = (
            dd.Normal(mean, std).log_prob(latents)
            - dd.Normal(0.0, 1.0).log_prob(latents)
            )
    aggregate = lambda x: tf.reduce_mean(tf.reduce_sum(x, 1))
    objective = aggregate(prior_divergence) - aggregate(log_likelihood)

    # Optimisation
    train_step = tf.train.AdamOptimizer(1e-3).minimize(objective)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for _ in range(1000):
        ix = np.random.choice(np.arange(1000), size=16, replace=False)
        x, x_test = data['train'][ix], data['test'][ix]
        print(sess.run(objective, feed_dict={feats: x_test}))
        sess.run(train_step, feed_dict={feats: x})

    # Successive image generations
    initial_image = data['test'][[15]]
    _, axes = plt.subplots(3, 4)
    for axes_ in axes:
        for axis in axes_:
            if axis != axes[0][0]:
                for _ in range(10):
                    initial_image = sess.run(preds, feed_dict={feats: initial_image})
            plot(axis, initial_image)
    plt.show()
