from torch.utils.data import Dataset
from os import listdir, path
import xarray as xr
from climatenet.utils.utils import Config
import random
import numpy as np

class ClimateDataset(Dataset):
    '''
    The basic Climate Dataset class. 

    Parameters
    ----------
    path : str
        The path to the directory containing the dataset (in form of .nc files)
    config : Config
        The model configuration. This allows to automatically infer the fields we are interested in 
        and their normalisation statistics

    Attributes
    ----------
    path : str
        Stores the Dataset path
    fields : dict
        Stores a dictionary mapping from variable names to normalisation statistics
    files : [str]
        Stores a sorted list of all the nc files in the Dataset
    length : int
        Stores the amount of nc files in the Dataset
    '''
  
    def __init__(self, path: str, config: Config):
        self.path: str = path
        self.fields: dict = config.fields
        self.augment: bool = config.augment
        self.files: [str] = [f for f in sorted(listdir(self.path)) if f[-3:] == ".nc"]
        self.length: int = len(self.files)
      
    def __len__(self):
        return self.length

    def normalize(self, features: xr.DataArray):
        for variable_name, stats in self.fields.items():   
            var = features.sel(variable=variable_name).values
            var -= stats['mean']
            var /= stats['std']

    def get_features(self, dataset: xr.Dataset):
        features = dataset[list(self.fields)].to_array()
        self.normalize(features)
        return features.transpose('time', 'variable', 'lat', 'lon')

    def __getitem__(self, idx: int):
        file_path: str = path.join(self.path, self.files[idx]) 
        dataset = xr.load_dataset(file_path)
        return self.get_features(dataset)

    @staticmethod
    def collate(batch):
        return xr.concat(batch, dim='time')

class ClimateDatasetLabeled(ClimateDataset):
    '''
    The labeled Climate Dataset class. 
    Corresponds to the normal Climate Dataset, but returns labels as well and batches accordingly
    ''' 
    def augment_features(self, features: xr.DataArray, random_longitude_shift: int):
        for variable_name, stats in self.fields.items():
            var = features.sel(variable=variable_name).values
            features.sel(variable=variable_name).values = np.roll(var, random_longitude_shift, axis=2)

    def augment_labels(self, labels: xr.DataArray, random_longitude_shift: int):
        pixels = labels.values
        labels.values = np.roll(pixels, random_longitude_shift, axis=1)

    def get_features_and_labels(self, dataset: xr.Dataset):
        
        # Normalize features
        features = dataset[list(self.fields)].to_array()
        self.normalize(features)

        # Augment features with random longitude shifts
        random_gamma = random.randint(0, 1)
        random_longitude_shift = random_gamma * random.randint(0, 359)
        self.augment_features(features, random_longitude_shift)

        # Augment labels with same randon longitude shifts
        labels = dataset['LABELS']
        self.augment_labels(labels, random_longitude_shift)
        
        return features.transpose('time', 'variable', 'lat', 'lon'), labels

    def __getitem__(self, idx: int):
        file_path: str = path.join(self.path, self.files[idx]) 
        dataset = xr.load_dataset(file_path)
        if self.augment:
            return self.get_features_and_labels(dataset)
        else:
            return self.get_features(dataset), dataset['LABELS']

    @staticmethod 
    def collate(batch):
        data, labels = map(list, zip(*batch))
        return xr.concat(data, dim='time'), xr.concat(labels, dim='time')