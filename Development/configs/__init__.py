"""This package includes all the modules related to data loading and preprocessing.
    To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
"""
import os
import json
from utils import load_configs
"""
def load_configs():
    configs = {}
    for pos_json in os.listdir('.'):
        if pos_json.endswith('.json'):
            with open('configs/' +pos_json) as json_file:
                for name, config in json.load(json_file).items():
                    if name in configs:
                        raise Exception(f"Config from {pos_json} with name {name} already exists!")
                    configs.update({name:config})
    return configs
"""
def load_config(name):
    return load_configs('configs/')[name]