def split_custom_filename(filename:str, sep:str='#') -> str:
    """Split filenames based on custom seperator.

    Args:
      filename(str): 
      sep(str, optional): (Default value = '#')

    Returns:

    Raises:

    """
    
    assert sep in filename, f"The expected seperator ({sep}) could not be found in filename"
    slices= filename.split(sep)
    slices[-1] = slices[-1].split(".")[0]

    return slices

def remove_preprocessed_filename_definition(filename:str) -> str:
    """Remove the three first letters to hyandle preprocessed names

    Args:
      filename(str): 

    Returns:

    Raises:

    """
    return filename[2:]

