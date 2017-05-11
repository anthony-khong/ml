class BernoulliDecoder(object):
    def __init__(self, n_latents, n_hidden, n_feats):
        self.n_latents = n_latents
        self.n_hidden = n_hidden
        self.n_feats = n_feats
