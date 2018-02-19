import os
import requests
from io import StringIO

import pandas as pd

import ml

def download_wine_quality():
    base_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'
    wine_quality_ext = 'wine-quality/winequality-red.csv'
    page = requests.get(base_url + wine_quality_ext)
    csv = page.text
    return csv

def load_wine_quality():
    expected_csv_path = ml.paths.dropbox() + '/datasets/wine_quality.csv'
    if not os.path.isfile(expected_csv_path):
        csv = download_wine_quality()
        df = pd.read_csv(StringIO(csv), sep=';')
        df.to_csv(expected_csv_path, index=False)
    return df.read_csv(expected_csv_path)
