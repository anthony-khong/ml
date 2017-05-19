import numpy as np
import tensorflow as tf

import reprod as rr

class VAE(object):
    def __init__(self, encoder, decoder, objective_fn, optimiser):
        self.encoder = encoder
        self.decoder = decoder
        self.objective_fn = objective_fn
        self.optimiser = optimiser

        self.sess = None
        self.history = []

    def train(self, data, n_epochs, minibatch_size, eval_fn=None, seed=None,
              verbose=True):
        if seed:
            np.random.seed(seed)
        eval_fn = eval_fn or (lambda model, data: None)
        if self.sess is None:
            self.sess = self.initialise_model()
        for i in range(n_epochs):
            self.train_for_one_epoch(data, minibatch_size)
            evaluation = eval_fn(self, data)
            self.history.append(evaluation)
            if verbose:
                print('Epoh: {}, Loss: {}'.format(i + 1, evaluation))

    def train_for_one_epoch(self, data, minibatch_size):
        n_examples = len(data['train'])
        epoch_ixs = rr.nnet.split_ixs_for_one_epoch(n_examples, minibatch_size)
        for ix in epoch_ixs:
            minibatch = data['train'][ix]
            self.sess.run(self.train_step, feed_dict={self.feats: minibatch})

    def initialise_model(self):
        self.feats = tf.placeholder(tf.float32, [None, self.encoder.n_feats])
        self.encoder_vars = self.encoder.forward_pass(self.feats)
        self.decoder_vars = self.decoder.forward_pass(self.encoder_vars)
        self.objective = self.objective_fn(self.encoder_vars, self.decoder_vars)
        self.train_step = self.optimiser.minimize(self.objective)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        return sess

    def eval_objective(self, data):
        objective = self.sess.run(self.objective, feed_dict={self.feats: data})
        return objective

    def sample(self, image):
        sess, feats, preds = self.sess, self.feats, self.decoder_vars['preds']
        new_image = sess.run(preds, feed_dict={feats: image})
        return new_image
