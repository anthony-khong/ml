from funcy import *

def reduce_values(reduce_fn, dictionaries, init_dict=None):
    for dictionary in dictionaries:
        if init_dict is None:
            init_dict = dictionary
        else:
            init_dict = {k: reduce_fn(v, dictionary[k]) for k, v in init_dict.items()}
    return init_dict

if __name__ == '__main__':
    import numpy as np

    stacker = curry(reduce_values)(np.append)
    results = stacker([
        {'x': [1, 2], 'y': [3, 4]},
        {'x': [5, 6], 'y': [7, 8]},
        ])
    print(results)

