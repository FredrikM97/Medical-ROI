import os
import nibabel as nib

def load_spm_data(path) -> iter:
    files = os.listdir(path)
    for file in files:  
        pet_img = nib.load(path+file).get_fdata() # Load image
        yield pet_img.T
        
def load_adni_data(path) -> iter:
    files = os.listdir(path)
    for file in files:  
        pet_img = nib.load(path+file).get_fdata() # Load image
        yield pet_img.T[0]
        
def load_spm_file(path):
    return nib.load(path).get_fdata().T

def load_adni_file(path):
    return nib.load(path).get_fdata().T[0]