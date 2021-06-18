"""
    This package includes all the modules related to data loading and preprocessing.
"""
import importlib
import sys

from pytorch_lightning import LightningDataModule
from torch.utils import data

from src.files import load

from . import *

BASEDIR = '.'.join([str(__name__),'dataloader'])

def find_dataset_using_name(datadir, dataset_name):
    """

    Args:
      datadir: 
      dataset_name: 

    Returns:

    Raises:

    """
    # datadir should be full path to file and dataset_name is the class within the file!
    dataset_filename = datadir
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '')
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() and issubclass(cls, LightningDataModule):
            dataset = cls

    if dataset is None:
        raise NotImplementedError('In {0}.py, there should be a subclass of BaseDataset with class name that matches {1} in lowercase.'.format(dataset_filename, target_dataset_name))

    return dataset

def create_dataset(name:str=None, args:dict={},**kwargs):
    """

    Args:
      name(str, optional): (Default value = None)
      args(dict, optional): (Default value = {})
      **kwargs: 

    Returns:

    Raises:

    """
    return find_dataset_using_name(BASEDIR,name)(**args, **kwargs)

