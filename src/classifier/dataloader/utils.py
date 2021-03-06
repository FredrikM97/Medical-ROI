from sklearn.model_selection import KFold
import os
import numpy as np

__all__ = ['get_labels','get_nii_files']

import os
def filename2labels(images:list, classes:dict, delimiter:str):
    assert delimiter in images[0], "The defined delimiter for ClassWeights could not be found in image input!"
    return np.array([classes[img_path.rsplit("/",1)[1].split(delimiter,1)[0]] for img_path in images])

def load_files(srcdir):
    return np.array([
            path + '/' + filename
            for path, _, files in os.walk(srcdir) 
            for filename in files if filename.endswith('.nii')
        ])