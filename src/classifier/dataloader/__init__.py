"""This package includes all the modules related to data loading and preprocessing.
    To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
"""
import importlib
from torch.utils import data
from pytorch_lightning import LightningDataModule
BASEDIR = 'neural_network.'


def create_dataset(**configuration:dict):
    """Create a dataset given the configuration (loaded from the json file).
    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and train.py/validate.py
    Example:
        from datasets import create_dataset
        dataset = create_dataset(configuration)
    """
    return find_dataset_using_name(configuration['dataset_name'])(**configuration)

