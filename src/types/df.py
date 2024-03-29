from typing import Union

import pandas as pd


def df_object2type(
    input_df:pd.DataFrame,
    types:dict={
        'float':[],
        'int':[],
        'str':[],
        'cat':[],
        'datetime':[],
    },
    show_output:bool=True):
    """

    Args:
      input_df(pd.DataFrame): 
      types(dict, optional): (Default value = {'float':[],'int':[],'str':[],'cat':[],'datetime':[],})
      show_output(bool, optional): (Default value = True)

    Returns:

    Raises:

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

def object2type(data:str) -> Union[int, float]:
    """Change the type of data.
    Check if data is a int, float otherwise a string/object

    Args:
      data(str): 

    Returns:

    Raises:

    """
    if data.replace('.', '', 1).lstrip('-').isdigit():
        if data.isdigit():
            return int(data)
        else:
            return float(data)
    return data

def xml2dict(r, parent:str='', delimiter:str=".") -> list:
    """Iterate through all xml files and add them to a dictionary

    Args:
      r: 
      parent(str, optional): (Default value = '')
      delimiter(str, optional): (Default value = ".")

    Returns:

    Raises:

    """
    param = lambda r,delimiter:delimiter+list(r.attrib.values())[0].replace(" ", "_") if r.attrib else ''
    def recursive(r:str, parent:str, delimiter:str='.') -> list:
        """

        Args:
          r(str): 
          parent(str): 
          delimiter(str, optional): (Default value = '.')

        Returns:

        Raises:

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

def column_to_tuple(pd_column:'pandas.DataFrame') -> 'pandas.DataFrame':
    """Convert a pandas column from string to tuple

    Args:
      pd_column('pandas.DataFrame'): Series

    Returns:
      type: output (Series):

    Raises:

    """
    
    return pd_column.apply(ast.literal_eval)

def column_to_np(pd_column:'pandas.DataFrame', dtype:str='float64') -> 'pandas.DataFrame':
    """Convert a pandas column from tuple to numpy arrays

    Args:
      pd_column('pandas.DataFrame'): Series
      dtype(str, optional): (Default value = 'float64')

    Returns:
      type: output (Series):

    Raises:

    """
    
    return pd_column.apply(lambda x: np.array(x, dtype=dtype))