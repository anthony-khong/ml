import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.contrib import distributions as dd

from reprod.datasets.mnist import load_mnist
from vae.decoder import BernoulliDecoder
from vae.encoder import GaussianEncoder
from vae.model import VAE

def objective_fn(encoder_vars, decoder_vars):
    feats, preds = encoder_vars['feats'], decoder_vars['preds']
    mean, std = encoder_vars['mean'], encoder_vars['std']
    latents = encoder_vars['latents']

    log_likelihood = feats*tf.log(preds) + (1.0-feats)*tf.log(1.0-preds)
    norm_lpdf = lambda m, s, x: dd.Normal(m, s).log_prob(x)
    prior_divergence = norm_lpdf(mean, std, latents) - norm_lpdf(0.0, 1.0, latents)
    aggregate = lambda x: tf.reduce_mean(tf.reduce_sum(x, 1))
    objective = aggregate(prior_divergence) - aggregate(log_likelihood)
    return objective

def eval_fn(model, data):
    evaluation = {s: model.eval_objective(x) for s, x in data.items()}
    return evaluation

def sample_images(model, initial_image, n_thins):
    def draw_digit(axis, pixels):
        pixels = np.array(255 * pixels, dtype='uint8').reshape((28, 28))
        axis.imshow(pixels, cmap='gray')

    def generate_new_image(image):
        for _ in range(n_thins):
            image = model.sample(image)
        return image

    image = None
    _, axes_list = plt.subplots(3, 4)
    axes = [axis for axes in axes_list for axis in axes]
    for i, axis in enumerate(axes):
        image = initial_image if i == 0 else generate_new_image(image)
        draw_digit(axis, image)
    plt.show()

if __name__ == '__main__':
    n_feats = 784
    n_hidden = 200
    n_latents = 20
    n_epochs = 5
    minibatch_size = 16

    data = load_mnist()
    encoder = GaussianEncoder(n_feats, n_hidden, n_latents)
    decoder = BernoulliDecoder(n_latents, n_hidden, n_feats)
    optimiser = tf.train.AdamOptimizer(1e-3)

    vae = VAE(encoder, decoder, objective_fn, optimiser)
    vae.train(data, n_epochs, minibatch_size, eval_fn)

    final_loss = eval_fn(vae, data)
    print('Final loss: {}'.format(final_loss))

    initial_image = data['test'][[100]]
    sample_images(vae, initial_image, n_thins=1)
