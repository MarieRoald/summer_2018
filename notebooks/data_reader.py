import pandas as pd

class DataReader:
    def __init__(self, data_set_filenames, groups_filename):
        
        self.data_set_filenames = data_set_filenames

        self.dataframes = [self._load_dataset(filename) for filename in self.data_set_filenames]
        self.groups = self._load_groups(groups_filename) 

    def _load_dataset(self, filename):
        data = pd.read_csv(filename, index_col=0)
        return data

    def _load_groups(self, filename):
        groups = pd.read_csv(filename, index_col=0, names=["Sample ID", "Person ID"])
        return groups

    def get_all_data(self):
        return pd.concat(self.dataframes, axis=1)

    def get_groups(self):
        return self.groups

    

