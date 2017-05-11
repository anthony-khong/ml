import tensorflow as tf
from reprod.datasets.mnist import load_mnist
from vae.decoder import BernoulliDecoder
from vae.encoder import GaussianEncoder
from vae.model import VAE

if __name__ == '__main__':
    n_feats = 784
    n_hidden = 100
    n_latents = 10
    n_epochs = 100
    minibatch_size = 16

    data = load_mnist()
    encoder = GaussianEncoder(n_feats, n_hidden, n_latents)
    decoder = BernoulliDecoder(n_latents, n_hidden, n_feats)
    def objective(encoder_vars, decoder_vars):
        pass
    optimiser = tf.train.AdamOptimizer(1e-3)

    vae = VAE(encoder, decoder, objective, optimiser)
    def eval_fn():
        pass
    vae.train(data['train'], n_epochs, minibatch_size, eval_fn)
