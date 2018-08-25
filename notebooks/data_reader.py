import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import numpy as np

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

    @property
    def combined_data(self):
        return pd.concat(self.dataframes, axis=1)

    @property
    def seperate_data(self):
        return self.dataframes


class MultimodalDataset:
    def __init__(self, datasets, groups=None):

        self.datasets = datasets
        self.groups = groups

    @property
    def combined_data(self):
        return pd.concat(self.datasets, axis=1)

    @property
    def seperate_data(self):
        return self.datasets

    @property
    def input_shapes(self):
        return [(d.shape[1],) for d in self.datasets]

    @property
    def shapes(self):
        return [d.shape for d in self.datasets]

    @property
    def n_samples(self):
        return self.datasets[0].shape[0]

    @property
    def lengths(self):
        lengths = [d.shape[1] for d in self.datasets]
        return lengths

    @property
    def seperators(self):
        separators = np.cumsum([0] + self.lengths)
        return separators

    def __getitem__(self, idx):
        return [d.iloc[idx] for d in self.datasets]

    def train_test_split(self, random_state=100):
        train_idx, val_idx = next(GroupShuffleSplit(random_state=random_state).split(self.datasets[0], groups=self.groups))
        data_train = self[train_idx]
        data_val = self[val_idx]
        return data_train, data_val

