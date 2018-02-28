import numpy as np

def minibatch_splitter(size):
    def split(*arrays):
        assert len(set(len(xs) for xs in arrays)) == 1, (
                'All arrays must have the same length in the first '
                'axis in order to be split into minibatches.'
                )
        n_rows = len(arrays[0])
        n_minibatches = np.ceil(n_rows / size).astype(int)
        random_ixs = np.random.permutation(np.arange(n_rows))
        for i in range(n_minibatches):
            start_ix, end_ix = i*size, (i+1)*size
            minibatch_ix = random_ixs[start_ix:end_ix]
            yield [xs[minibatch_ix] for xs in arrays]
    return split

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

if __name__ == '__main__':
    split = minibatch_splitter(3)
    for x, y in split(np.arange(100).reshape(10, 10).T, np.arange(30).reshape(3, 10)):
        print(x, '\n', y, '\n')
