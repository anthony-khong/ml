from abc import ABCMeta, abstractmethod

class DataSet(metaclass=ABCMeta):
    @abstractmethod
    def download(self):
        pass

class UCIDataSet(DataSet):
    base_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'
