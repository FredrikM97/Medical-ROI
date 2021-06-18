import os

def create_directory(dest:str)-> bool:
    """Creates directory if not exists

    Args:
      dest(str): Destination path to create directory

    Returns:

    Raises:

    """
    return os.makedirs(os.path.dirname(dest), exist_ok=True)

def filter_dir(in_dir:str, out_dir:str) -> None:
    """Remove all but 1 .nii file from each bottom-level folder, keeps in_dir intact and puts result in out_dir. Example usage: data_filter('..\\data\\adni_raw', '..\\data\\adni_raw_filtered')

    Args:
      in_dir(str): Source directory
      out_dir(str): Output directory

    Returns:

    Raises:

    """
    ""
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
        
def copy(src:str, dest:str)-> bool:
    """Copy file from source dir to dest dir. Note that the path must exist where folders should be placed!

    Args:
      src(str): Path to source
      dest(str): Path to destination

    Returns:

    Raises:

    """
    assert os.path.exists(src), "Source file does not exists"
    
    if not os.path.exists(dest):
        shutil.copy(src, dest)
        return True
    return False


def availible_files(path:str, contains:str='') -> list:
    """Returns the availible files in directory

    Args:
      path(str): Path to directory
      contains(str, optional): (Default value = '')

    Returns:

    Raises:

    """
    return [f for f in os.listdir(path) if contains in f]

def write(filepath:str, message:str, flag:str='a') -> None:
    """Write to a file defined by a filepath and a given message

    Args:
      filepath(str): Path to file
      message(str): Message to write to file
      flag(str, optional): (Default value = 'a')

    Returns:

    Raises:

    """
    try:
        with open(filepath, flag) as f:
            f.write(message)
    except Exception as e:
        raise Exception("Could not write message to file!") from e