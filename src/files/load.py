"""
Predefined loading of different formats to python.
"""


import importlib
import itertools
import json
import os
import xml.etree.ElementTree as ET
from typing import Dict

import nibabel as nib
import numpy as np

from .preprocess import image2axial


def load_json(dirpath:str=None) -> dict:
    """Open a json file and return the key name and config data if the file exists

    Args:
      dirpath(str, optional): path to json file (Default value = None)

    Returns:
      dict: * Dictionary with {name:config}

    Raises:
      ValueError If file could not be loaded

    """
    with open(dirpath) as json_file:
        name = ''
        try:
            return {name:config for name, config in json.load(json_file).items()}
        except ValueError as e:
            raise ValueError(f" Error when loading file: {dirpath}") from e

def load_config(filename:str,dirpath:str=None) -> Dict:
    """Load file with same name as the input filename and located in dirpath

    Args:
      filename(str): 
      dirpath(str, optional): (Default value = None)

    Returns:

    Raises:

    """

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
    """Load XML from dictory and return a generator

    Args:
      path(str): 

    Returns:

    Raises:

    """
    assert path, "No path defined"
    for filename in os.listdir(path):
        if not filename.endswith('.xml'): continue
        fullname = os.path.join(path, filename)
        yield ET.parse(fullname)
        
def load_nifti(srcdir:str) -> list:
    """Load nifti image from srcdir where all files ending with .nii is selected.

    Args:
      srcdir(str): path to source

    Returns:
      : List of dictionaries containing columns, filename and path to file

    Raises:

    """
    columns = ['filename','dirs','path']
    return [
            dict(
                zip(columns,[filename,path])
            ) 
            for path, _, files in os.walk(srcdir) 
            for filename in files if filename.endswith('.nii')
        ] 

def files_path(srcdir:str) -> 'np.ndarray':
    """Load a file from srcdir if the file ends with .nii

    Args:
      srcdir(str): 

    Returns:

    Raises:

    """
    tmp = np.array([
            path + '/' + filename
            for path, _, files in os.walk(srcdir) 
            for filename in files if filename.endswith('.nii')
        ])
    if len(tmp) == 0: raise ValueError(f"No files loaded from path {srcdir} that ends with extension .nii")

    return tmp

def nifti_axial(path:str) -> 'np.ndarray':
    """Load an nifti image from the axial view

    Args:
      path(str): path to the given file

    Returns:
      np.ndarray: * Nifti image from axial view

    Raises:

    """
    # Load nifti image and convert to axial view
    return image2axial(nib.load(path).get_fdata())
     
def adni(path:str) -> iter:
    """Load a generator of all images or image if path points at a file

    Args:
      path(str): 

    Returns:

    Raises:

    """
    if os.path.isdir(path):
        files = os.listdir(path)
        for file in files:  
            #pet_img =  # Load image
            yield nifti_axial(path+file)#[0]
    else:
        return nifti_axial(path)#[0]
        
def spm(path:str):
    """Load a generator of all images or image if path points at a file

    Args:
      path(str): 

    Returns:

    Raises:

    """
    if os.path.isdir(path):
        files = os.listdir(path)
        for file in files:  
            yield nifti_axial(path+file)
    else:
        return nifti_axial(path)