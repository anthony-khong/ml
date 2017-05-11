class GaussianEncoder(object):
    def __init__(self, n_feats, n_hidden, n_latents):
        self.n_feats = n_feats
        self.n_hidden = n_hidden
        self.n_latents = n_latents
