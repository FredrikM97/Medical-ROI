"""
Utility function that are commonly used in different modules.
"""


import numpy as np
import pandas as pd
import os
import shutil
from typing import List,Dict
import xml.etree.ElementTree as ET

def availible_files(path, contains:str=''):
    """Returns the availible files in directory

    Parameters
    ----------
    path :
        
    contains : str
        (Default value = '')

    Returns
    -------

    
    """
    return [f for f in os.listdir(path) if contains in f]
            
def merge_dict(a, b, path:str=None):
    """

    Parameters
    ----------
    a :
        
    b :
        
    path : str
        (Default value = None)

    Returns
    -------

    
    """
    "merges b into a"
    if path is None: path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge_dict(a[key], b[key], path + [str(key)])
            else:
                a[key] = b[key]
        else:
            a[key] = b[key]
    return a

def df_object2type(
    input_df,
    types={
        'float':[],
        'int':[],
        'str':[],
        'cat':[],
        'datetime':[],
    },
    show_output:bool=True):
    """

    Parameters
    ----------
    input_df :
        
    types :
        (Default value = {'float':[],'int':[],'str':[],'cat':[],'datetime':[],})
    show_output : bool
        (Default value = True)

    Returns
    -------

    
    """
    
    converter = {
        'float':'float',
        'int':'int',
        'str':'string',
        'cat':'category',
        'datetime':'datetime64[ns]',
    }
    """Convert the type of a dataframe into the defined type categories."""
    
    if len(types.values()) and show_output:
        print("Processing type of:")
        
    for key,cols in types.items(): # get types
        t = converter.get(key)
        for col in cols:
            if show_output: print(f"\t {t}: {col}")
            input_df[col] =  input_df[col].astype(t)
            
    return input_df

def copy_file(src:str, dest:str)-> bool:
    """Copy file from source dir to dest dir. Note that the path must exist where folders should be placed!

    Parameters
    ----------
    src : str
        
    dest : str
        

    Returns
    -------

    
    """
    assert os.path.exists(src), "Source file does not exists"
    
    if not os.path.exists(dest):
        shutil.copy(src, dest)
        return True
    return False

def create_directory(dest)-> bool:
    """Creates directory if not exists

    Parameters
    ----------
    dest :
        

    Returns
    -------

    
    """
    return os.makedirs(os.path.dirname(dest), exist_ok=True)

def remove_preprocessed_filename_definition(filename:str):
    """remove the three first letters to hyandle preprocessed names

    Parameters
    ----------
    filename : str
        

    Returns
    -------

    
    """
    return filename[2:]


        
def xml2dict(r, parent:str='', delimiter:str=".") -> list:
    """Iterate through all xml files and add them to a dictionary

    Parameters
    ----------
    r :
        
    parent : str
        (Default value = '')
    delimiter : str
        (Default value = ".")

    Returns
    -------

    
    """
    param = lambda r,delimiter:delimiter+list(r.attrib.values())[0].replace(" ", "_") if r.attrib else ''
    def recursive(r, parent:str, delimiter:str='.') -> list:
        """

        Parameters
        ----------
        r :
            
        parent : str
            
        delimiter : str
            (Default value = '.')

        Returns
        -------

        
        """
        cont = {}
        # If list
        if (layers := r.findall("./*")):
            [cont.update(recursive(x, parent +delimiter+ x.tag)) for x in layers]
            return cont

        elif r.text and '\n' not in r.text: # get text
            return {parent + param(r,delimiter):object2type(r.text)}
        else:
            return {}
    return recursive(r, parent, delimiter=delimiter)

def object2type(data:str):
    """Change the type of data.
    Check if data is a int, float otherwise a string/object

    Parameters
    ----------
    data : str
        

    Returns
    -------

    
    """
    if data.replace('.', '', 1).lstrip('-').isdigit():
        if data.isdigit():
            return int(data)
        else:
            return float(data)
    return data


def split_custom_filename(filename:str, sep:str='#'):
    """

    Parameters
    ----------
    filename : str
        
    sep : str
        (Default value = '#')

    Returns
    -------

    
    """
    "Split filenames based on custom seperator."
    assert sep in filename, f"The expected seperator ({sep}) could not be found in filename"
    slices= filename.split(sep)
    slices[-1] = slices[-1].split(".")[0]

    return slices

def data_filter(in_dir:str, out_dir:str):
    """

    Parameters
    ----------
    in_dir : str
        
    out_dir : str
        

    Returns
    -------

    
    """
    "Will remove all but 1 .nii file from each bottom-level folder, keeps in_dir intact and puts result in out_dir. Example usage: data_filter('..\\data\\adni_raw', '..\\data\\adni_raw_filtered')"
    files_to_create = []
    for root, subdirs, files in os.walk(in_dir):
        nii_files_in_directory = []
        for file in files:
            if file.endswith('.nii'):
                nii_files_in_directory.append(file)
        if len(nii_files_in_directory) > 0:
            files_to_create.append([root.replace(in_dir, '', 1), nii_files_in_directory[0]])
    
    for file in files_to_create:
        assert create_directory(out_dir + file[0] + '\\'), "Could not create directory"
        assert copy_file(in_dir + file[0] + '\\' + file[1], out_dir + file[0] + '\\' + file[1]), "Source file already exists"