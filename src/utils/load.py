import os
import xml.etree.ElementTree as ET
from typing import Dict
import json
import importlib
import numpy as np
from src.utils.preprocess import image2axial
import nibabel as nib

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
    for pos_json in os.listdir(dirpath):
        name, extension = pos_json.rsplit(".",1)
        
        if  name == filename and extension == 'json': 
            return load_json(dirpath +'/'+pos_json)

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
<<<<<<< HEAD
    return tmp
=======
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
>>>>>>> Bug fixes and cleanup
