import tensorflow as tf

import reprod as rr

class GaussianEncoder(object):
    def __init__(self, n_feats, n_hidden, n_latents):
        self.n_feats = n_feats
        self.n_hidden = n_hidden
        self.n_latents = n_latents

    def forward_pass(self, feats):
        n_feats, n_hidden, n_latents = self.n_feats, self.n_hidden, self.n_latents
        make_weight, make_bias = rr.nnet.make_weight, rr.nnet.make_bias

        weights = {
                'hidden': make_weight((n_feats, n_hidden)),
                'mean': make_weight((n_hidden, n_latents)),
                'std': make_weight((n_hidden, n_latents))
                }
        biases = {
                'hidden': make_bias((n_hidden)),
                'mean': make_bias((n_latents)),
                'std': make_bias((n_latents))
                }

        hidden = tf.matmul(feats, weights['hidden']) + biases['hidden']
        mean = tf.matmul(hidden, weights['mean']) + biases['mean']
        std = tf.exp(tf.matmul(hidden, weights['std']) + biases['std'])
        noise = tf.random_normal([tf.shape(feats)[0], n_latents])
        latents = mean + noise*std

        self.weights = weights
        self.biases = biases
        self.encoder_vars = {
                'feats': feats,
                'hidden': hidden,
                'mean': mean,
                'std': std,
                'noise': noise,
                'latents': latents
                }
        return self.encoder_vars

