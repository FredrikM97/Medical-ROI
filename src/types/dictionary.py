def merge_dict(a, b, path:str=None):
    """

    Parameters
    ----------
    a :
        
    b :
        
    path : str
        (Default value = None)

    Returns
    -------

    
    """
    "merges b into a"
    if path is None: path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge_dict(a[key], b[key], path + [str(key)])
            else:
                a[key] = b[key]
        else:
            a[key] = b[key]
    return a




