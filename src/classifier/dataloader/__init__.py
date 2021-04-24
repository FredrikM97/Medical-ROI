"""This package includes all the modules related to data loading and preprocessing.
    To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
"""
import importlib
from torch.utils import data
from pytorch_lightning import LightningDataModule
from src.utils import load
from . import *
import sys
BASEDIR = '.'.join([str(__name__),'dataloader'])

def find_dataset_using_name(datadir, dataset_name):
    # datadir should be full path to file and dataset_name is the class within the file!
    dataset_filename = datadir
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '')
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, LightningDataModule):
            dataset = cls

    if dataset is None:
        raise NotImplementedError('In {0}.py, there should be a subclass of BaseDataset with class name that matches {1} in lowercase.'.format(dataset_filename, target_dataset_name))

    return dataset

def create_dataset(train_files, val_files, name:str=None, args:dict={}):
    return find_dataset_using_name(BASEDIR,name)(train_files, val_files, **args)

