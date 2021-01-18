import os
import xml.etree.ElementTree as ET
from functools import reduce
import shutil

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
    assert path, "No path defined"
    for filename in os.listdir(path):
        if not filename.endswith('.xml'): continue
        fullname = os.path.join(path, filename)
        tree = ET.parse(fullname)
        yield tree
        
def copy_file(src, dest):
    "Copy file from source dir to dest dir. Note that the path must exist where folders should be placed!"
    if not os.path.exists(dest):
        shutil.copy(src, dest)
        return 'ok' 
    return "fail"