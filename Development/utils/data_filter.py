import os
from .misc_util import copy_file, create_directory

def data_filter(in_dir, out_dir):
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