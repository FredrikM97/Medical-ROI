import os
from shutil import copy

def data_filter(in_dir, out_dir):
    "Will remove all but 1 .nii file from each bottom-level folder, keeps in_dir intact and puts result in out_dir. Example usage: data_filter('..\\data\\adni_raw', '..\\data\\adni_raw_filtered')"
    files_to_create = []
    for root, subdirs, files in os.walk(in_dir):
        files_in_directory = []
        for file in files:
            if file.endswith('.nii'):
                files_in_directory.append(file)
        if len(files) > 0:
            files_to_create.append([root.replace(in_dir, '', 1), files[0]])
    
    for file in files_to_create:
        os.makedirs(os.path.dirname(out_dir + file[0] + '\\'), exist_ok=True)
        copy(in_dir + file[0] + '\\' + file[1], out_dir + file[0] + '\\' + file[1])