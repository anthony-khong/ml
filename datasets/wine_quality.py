import requests

from reprod.datasets.base import UCIDataSet

class WineQuality(UCIDataSet):
    wine_quality_ext = 'wine-quality/winequality-red.csv'
    csv_url = UCIDataSet.base_url + wine_quality_ext

    def download(self):
        page = requests.get(self.csv_url)
        csv = page.text
        return csv
