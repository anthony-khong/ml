import os

def home():
    return os.path.expanduser('~')

def downloads():
    return home() + '/Downloads'

def dropbox():
    return home() + '/Dropbox'
