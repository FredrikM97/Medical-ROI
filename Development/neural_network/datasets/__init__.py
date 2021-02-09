"""This package includes all the modules related to data loading and preprocessing.
    To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
"""
import importlib
from torch.utils import data
from pytorch_lightning import LightningDataModule

def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".
    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "datasets." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, LightningDataModule):
            dataset = cls

    if dataset is None:
        raise NotImplementedError('In {0}.py, there should be a subclass of BaseDataset with class name that matches {1} in lowercase.'.format(dataset_filename, target_dataset_name))

    return dataset

def create_dataset(configuration:dict):
    """Create a dataset given the configuration (loaded from the json file).
    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and train.py/validate.py
    Example:
        from datasets import create_dataset
        dataset = create_dataset(configuration)
    """
    return find_dataset_using_name(configuration['dataset_name'])(**configuration)

