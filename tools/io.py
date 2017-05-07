import _pickle as pickle

def write(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def read(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj
