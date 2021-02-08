import torch
#from .config import NNConfig
import os

def get_nii_files(srcdir):
    return [
            path + '/' + filename
            for path, _, files in os.walk(srcdir) 
            for filename in files if filename.endswith('.nii')
        ] 

def do_print(string):
    print(string)
    
def label_encoder(labels):
    "Convert list of string labels to tensors"
    from sklearn import preprocessing
    import torch
    le = preprocessing.LabelEncoder()
    targets = le.fit_transform(labels)
                  
    return torch.as_tensor(targets)

def get_availible_files(path, contains:str=''):
    return [f for f in os.listdir(path) if contains in f]