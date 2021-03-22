import os
import xml.etree.ElementTree as ET
from typing import Dict
import json
import importlib
import numpy as np

def load_configs(dirpath:str=None) -> Dict:
    # Expects at least one file that ends with .json
    configs = {}
    dirpath = os.path.abspath(dirpath if dirpath else '.')
    for pos_json in os.listdir(dirpath):
        if pos_json.endswith('.json'):
            with open(dirpath +'/'+pos_json) as json_file:
                for name, config in json.load(json_file).items():
                    if name in configs:
                        raise Exception(f"Config from {pos_json} with name {name} already exists!")

                    configs.update({name:config})
    return configs

def load_xml(path:str):
    "Load XML from dictory and return a generator"
    assert path, "No path defined"
    for filename in os.listdir(path):
        if not filename.endswith('.xml'): continue
        fullname = os.path.join(path, filename)
        yield ET.parse(fullname)
        
def load_nifti(srcdir):
    columns = ['filename','dirs','path']
    return [
            dict(
                zip(columns,[filename,path])
            ) 
            for path, _, files in os.walk(srcdir) 
            for filename in files if filename.endswith('.nii')
        ] 

def load_files(srcdir:str):
    tmp = np.array([
            path + '/' + filename
            for path, _, files in os.walk(srcdir) 
            for filename in files if filename.endswith('.nii')
        ])
    if len(tmp) == 0: raise ValueError(f"No files loaded from path {srcdir} that ends with extension .nii")
    return tmp