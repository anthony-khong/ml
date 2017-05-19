import tensorflow as tf

import reprod as rr

class BernoulliDecoder(object):
    def __init__(self, n_latents, n_hidden, n_feats):
        self.n_latents = n_latents
        self.n_hidden = n_hidden
        self.n_feats = n_feats

    def forward_pass(self, encoder_vars):
        n_feats, n_hidden, n_latents = self.n_feats, self.n_hidden, self.n_latents
        make_weight, make_bias = rr.nnet.make_weight, rr.nnet.make_bias
        latents = encoder_vars['latents']

        weights = [make_weight((n_latents, n_hidden)),
                   make_weight((n_hidden, n_feats))]
        biases = [make_bias((n_hidden)), make_bias((n_feats))]

        hidden = tf.nn.relu6(tf.matmul(latents, weights[0]) + biases[0])
        preds = tf.sigmoid(tf.matmul(hidden, weights[1]) + biases[1])

        self.weights = weights
        self.biases = biases
        self.decoder_vars = {'hidden': hidden, 'preds': preds}
        return self.decoder_vars
