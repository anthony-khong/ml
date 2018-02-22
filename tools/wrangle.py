import numpy as np

def get_index_map(values):
    indices = {}
    for index, value in enumerate(values):
        if value in indices:
            indices[value].append(index)
        else:
            indices[value] = [index]
    return indices

def one_hot(values):
    index_map = get_index_map(values)
    def binary_vector(ixs):
        vector = np.repeat(0.0, len(values))
        vector[ixs] = 1.0
        return vector
    vectors = [binary_vector(indices) for indices in index_map.values()]
    return np.array(vectors).T

