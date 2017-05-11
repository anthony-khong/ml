import tensorflow as tf

class VAE(object):
    def __init__(self, encoder, decoder, objective, optimiser):
        self.encoder = encoder
        self.decoder = decoder
        self.objective = objective
        self.optimiser = optimiser

        self.sess = None
        self.history = []

    def train(self, data, n_epochs, minibatch_size, eval_fn=None):
        if self.sess is None:
            self.sess = self.initialise_session()
        for _ in range(n_epochs):
            self.train_for_one_epoch(data, minibatch_size)
            self.history.append(eval_fn(self, data))

    def train_for_one_epoch(self, data, minibatch_size):
        pass # TODO

    def initialise_session(self):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        return sess

