"""
Predefined loading of different formats to python.
"""


import os
import xml.etree.ElementTree as ET
from typing import Dict
import json
import importlib
import numpy as np
from src.utils.preprocess import image2axial
import nibabel as nib
import itertools

def load_json(dirpath:str=None) -> dict:
    """Open a json file and return the key name and config data if the file exists
    
    Args:
        * dirpath: path to json file.
        
    Return:
        * Dictionary with {name:config}
        
    Exception:
        * ValueError: If file could not be loaded
    """
    with open(dirpath) as json_file:
        name = ''
        try:
            return {name:config for name, config in json.load(json_file).items()}
        except ValueError as e:
            raise ValueError(f" Error when loading file: {dirpath}") from e

def load_config(filename:str,dirpath:str=None) -> Dict:
    """Load file with same name as the input filename and located in dirpath"""

    dirpath = os.path.abspath(dirpath if dirpath else '.')
    files = itertools.chain.from_iterable(itertools.starmap(lambda root,dirs, files: [*map(lambda f: os.path.join(root, f), filter(lambda x: '.json' in x and 'checkpoint' not in x,files))],os.walk('../conf')))
    content = [*filter(lambda file: filename in load_json(f"{file}"), files)]
    if len(content) == 1:
        return load_json(f"{content[0]}")[filename]
    elif len(content) > 1:
        raise ValueError(f"More than one config exists with name {filename}, files: {content}")
    else:
        raise ValueError(f"Could not find json file: {filename} in {dirpath}!")
    
        

def load_xml(path:str):
    """Load XML from dictory and return a generator"""
    assert path, "No path defined"
    for filename in os.listdir(path):
        if not filename.endswith('.xml'): continue
        fullname = os.path.join(path, filename)
        yield ET.parse(fullname)
        
def load_nifti(srcdir:str) -> list:
    """Load nifti image from srcdir where all files ending with .nii is selected.
    
    Args:
        * srcdir: path to .nii files
    
    Return:
        * List of dictionaries containing columns, filename and path to file
        
    """
    columns = ['filename','dirs','path']
    return [
            dict(
                zip(columns,[filename,path])
            ) 
            for path, _, files in os.walk(srcdir) 
            for filename in files if filename.endswith('.nii')
        ] 

def load_files(srcdir:str):
    """Load a file from srcdir if the file ends with .nii"""
    tmp = np.array([
            path + '/' + filename
            for path, _, files in os.walk(srcdir) 
            for filename in files if filename.endswith('.nii')
        ])
    if len(tmp) == 0: raise ValueError(f"No files loaded from path {srcdir} that ends with extension .nii")

    return tmp

def load_nifti_axial(path:str) -> np.ndarray:
    """Load an nifti image from the axial view
    
    Args:
        * path: path to the given file
        
    Return:
        * Nifti image from axial view
    """
    # Load nifti image and convert to axial view
    return image2axial(nib.load(path).get_fdata())
