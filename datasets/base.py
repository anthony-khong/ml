from abc import ABCMeta, abstractmethod

class DataSet(metaclass=ABCMeta):
    @abstractmethod
    def download(self):
        pass

    @abstractmethod
    def get_df(self):
        pass

    def write_df(self, path):
        df = self.get_df()
        df.to_hdf(path, 'df')

class UCIDataSet(DataSet):
    base_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'
