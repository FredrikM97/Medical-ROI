import os
import xml.etree.ElementTree as ET
from functools import reduce
import shutil
import nibabel as nib
"""
class ADNI:
    columns = [
        'projectIdentifier',
        'subject.subjectIdentifier',
        'ImageProtocol.description',
        'dateAcquired',
        'subject.study.imagingProtocol.imageUID', 
        'filename',
        'path'
    ]
    files = []
    
    def __init__(self, data_dir, derivatives=True):
        self.data_dir = data_dir
        self.derivatives = derivatives
    
    def columns(self):
        return self.columns
    
    def load(self, dir):
        files = nib.load(dir)
        self.files = files
    def get(self):
        #ADNI_002_S_0295_PT_ADNI_Brain_PET__Raw_FDG_br_raw_20110609102421118_60_S111104_I239487.nii
        #
        pass
    
    def to_df(self):
        pass
       
    def nii_header(self,nib_image):
        pass
"""

def get_images(rootdir=None)-> dict:
    """Get all files from folder and subfolder of nii images. Based on ADNI directory paths!"""
    columns=[
        'projectIdentifier',
        'subject.subjectIdentifier',
        'ImageProtocol.description',
        'dateAcquired',
        'subject.study.imagingProtocol.imageUID', 
        'filename',
        'path'
    ]
    
    assert rootdir, "No rootdir selected"
    contents = []
    temp_dir = {}
    start = rootdir.rfind(os.sep) + 1
    for path, dirs, files in os.walk(rootdir):
        folders = path[start:].split(os.sep)
        subdir = dict.fromkeys(files)
        contents.extend((*folders,file, path + "/"+file) for file in files)

        #parent = reduce(dict.get, folders[:-1], temp_dir)
        #parent[folders[-1]] = subdir

    return contents, columns

def get_XML(path=None) -> iter:
    "Load XML from dictory and return a generator"
    assert path, "No path defined"
    for filename in os.listdir(path):
        if not filename.endswith('.xml'): continue
        fullname = os.path.join(path, filename)
        tree = ET.parse(fullname)
        yield tree
        
def copy_file(src, dest):
    "Copy file from source dir to dest dir. Note that the path must exist where folders should be placed!"
    assert os.path.exists(src), "Source file does not exists"
    
    if not os.path.exists(dest):
        shutil.copy(src, dest)
        return 'ok' 
    return "fail"

def save_to_categorised_images(output_df, base_dir)->None:
    "Save images to categories based on parameters from dataframe"
    output_df.apply(lambda row: copy_file(str(row['path']), f"{base_dir}/{row['subject.researchGroup']}/{row['filename']}"), axis=1)
    
def load_nii_data(path) -> iter:
    files = os.listdir(path)
    for file in files:
        if file[-4:] == '.nii':
            pet_img = nib.load(path+file).get_fdata() # Load image
            yield pet_img.T[0] # Shape: (z,x,y)
            
def load_spm_data(path) -> iter:
    files = os.listdir(path)
    for file in files:
        if file[-4:] == '.nii':
            pet_img = nib.load(path + file).get_fdata()
            yield pet_img.T