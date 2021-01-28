import pandas as pd
import nibabel as nib
import yaml
import os
import shutil
import sys
import types
from typing import List,Dict
import xml.etree.ElementTree as ET

def list_to_pandas(input_list, columns=None):
    return pd.DataFrame(input_list,columns=columns)

def split_dataset(files:list, split_ratio={'train':0.70,'validation':0.15, 'test':0.15})-> (list,list,list):
    DATASET_SIZE = len(files) 
    set_sizes = {
        'train':int(DATASET_SIZE*split_ratio['train']),
        'validation':int(DATASET_SIZE*split_ratio['validation']),
        'test':int(DATASET_SIZE*split_ratio['test'])
    }
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2

    
    return train_set, validation_set, test_set


def convert_df_types(
    input_df,
    types={
        'float':[],
        'int':[],
        'str':[],
        'cat':[],
        'datetime':[],
    },
    show_output=True):
    
    converter = {
        'float':'float',#pd.to_numeric,
        'int':'int',
        'str':'string',#pd.to_string,
        'cat':'category',#pd.Categorical,
        'datetime':'datetime64[ns]',#pd.to_datetime,
    }
    if len(types.values()) and show_output:
        print("Processing type of:")
    for key,cols in types.items(): # get types
        t = converter.get(key)
        for col in cols:
            if show_output: print(f"\t {t}: {col}")
            input_df[col] =  input_df[col].astype(t)
            
    return input_df

def copy_file(src, dest)-> bool:
    "Copy file from source dir to dest dir. Note that the path must exist where folders should be placed!"
    assert os.path.exists(src), "Source file does not exists"
    
    if not os.path.exists(dest):
        shutil.copy(src, dest)
        return True
    return False

def create_directory(dest)-> bool:
    "Creates directory if not exists"
    return os.makedirs(os.path.dirname(dest), exist_ok=True)

def get_nii_files(srcdir):
    columns = ['filename','dirs','path']
    return [
            dict(
                zip(columns,[filename,path])
            ) 
            for path, _, files in os.walk(srcdir) 
            for filename in files if filename.endswith('.nii')
        ] 
def remove_preprocessed_filename_definition(filename):
    "remove the three first letters to hyandle preprocessed names"
    return filename[2:]

def absolute_path(path):
    return os.path.abspath(path)

def load_xml(path:str):
    "Load XML from dictory and return a generator"
    assert path, "No path defined"
    for filename in os.listdir(path):
        if not filename.endswith('.xml'): continue
        fullname = os.path.join(path, filename)
        yield ET.parse(fullname)
        
def xml_to_dict(r, parent='', delimiter=".") -> list:
    "Iterate through all xml files and add them to a dictionary"
    param = lambda r:"_"+list(r.attrib.values())[0] if r.attrib else ''
    def recursive(r, parent='', delimiter=".") -> list:
        cont = {}
        # If list
        if layers := r.findall("./*"):
            [cont.update(recursive(x, parent=parent +delimiter+ x.tag)) for x in layers]
            return cont

        elif r.text and '\n' not in r.text: # get text
            return {parent + param(r):r.text}
        else:
            return {}
    return recursive(r, parent=parent, delimiter=delimiter)

def split_custom_filename(filename:str, sep='#'):
        "Split filenames based on custom seperator."
        slices= filename.split(sep)
        slices[-1] = slices[-1].split(".")[0]

        return slices
    
def load_image(path:str):
    "Load one image"
    return nib.load(path)

def load_images(files:List[str]) -> iter:
    "Load image file into memory. Expects a list of dict which contains path to file"
    func = lambda file: load_image(file)
    
    return generator(files, func)

def generator(content:list, func:types.FunctionType=None) -> iter:
    "Universal generator. Takes a list and a function that should be yielded."
    if not func: func = lambda x: x
        
    for c in content:
        yield func(c)
        
def default_print(inpute:str)->bool:
    print(inpute)
    return True
    
def merge_df(one_df,two_df, cols:list=None):
    """Merge two dataframes where columns are equal'"""
    return one_df.merge(two_df,on=cols)