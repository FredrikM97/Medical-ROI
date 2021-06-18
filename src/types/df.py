import pandas as pd

def df_object2type(
    input_df,
    types={
        'float':[],
        'int':[],
        'str':[],
        'cat':[],
        'datetime':[],
    },
    show_output:bool=True):
    """

    Parameters
    ----------
    input_df :
        
    types :
        (Default value = {'float':[],'int':[],'str':[],'cat':[],'datetime':[],})
    show_output : bool
        (Default value = True)

    Returns
    -------

    
    """
    
    converter = {
        'float':'float',
        'int':'int',
        'str':'string',
        'cat':'category',
        'datetime':'datetime64[ns]',
    }
    """Convert the type of a dataframe into the defined type categories."""
    
    if len(types.values()) and show_output:
        print("Processing type of:")
        
    for key,cols in types.items(): # get types
        t = converter.get(key)
        for col in cols:
            if show_output: print(f"\t {t}: {col}")
            input_df[col] =  input_df[col].astype(t)
            
    return input_df

def object2type(data:str):
    """Change the type of data.
    Check if data is a int, float otherwise a string/object

    Parameters
    ----------
    data : str
        

    Returns
    -------

    
    """
    if data.replace('.', '', 1).lstrip('-').isdigit():
        if data.isdigit():
            return int(data)
        else:
            return float(data)
    return data

def xml2dict(r, parent:str='', delimiter:str=".") -> list:
    """Iterate through all xml files and add them to a dictionary

    Parameters
    ----------
    r :
        
    parent : str
        (Default value = '')
    delimiter : str
        (Default value = ".")

    Returns
    -------

    
    """
    param = lambda r,delimiter:delimiter+list(r.attrib.values())[0].replace(" ", "_") if r.attrib else ''
    def recursive(r, parent:str, delimiter:str='.') -> list:
        """

        Parameters
        ----------
        r :
            
        parent : str
            
        delimiter : str
            (Default value = '.')

        Returns
        -------

        
        """
        cont = {}
        # If list
        if (layers := r.findall("./*")):
            [cont.update(recursive(x, parent +delimiter+ x.tag)) for x in layers]
            return cont

        elif r.text and '\n' not in r.text: # get text
            return {parent + param(r,delimiter):object2type(r.text)}
        else:
            return {}
    return recursive(r, parent, delimiter=delimiter)
