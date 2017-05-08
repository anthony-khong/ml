import numpy as np

def center(x):
    if x.ndim == 1:
        return x - np.mean(x)
    elif x.ndim == 2:
        return x - np.mean(x, axis=0)
    else:
        raise NotImplementedError

def standardise(x):
    if x.ndim == 1:
        return x / np.std(x)
    elif x.ndim == 2:
        sigma = np.std(x, axis=0)
        sigma[sigma==0] = 1.0
        return x / sigma
    else:
        raise NotImplementedError
