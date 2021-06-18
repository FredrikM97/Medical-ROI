"""
Common print functions used in different modules.
"""



import pandas as pd
import sys
from ruamel.yaml import YAML
import matplotlib.pyplot as plt
import matplotlib as mpl
from math import ceil,floor
import seaborn as sns

def pd_cols(input_df:pd.DataFrame) -> None:
    """Plot all columns of a pandas dataframe

    Parameters
    ----------
    input_df :
        

    Returns
    -------

    
    """
    "Print all pandas columns"
    with pd.option_context('display.max_columns', None):
        display(input_df.head())

def dict2yaml(input_dict:dict) -> None:
    """Plot a dictionary in yaml format.

    Parameters
    ----------
    input_dict : dict
        

    Returns
    -------

    
    """
    "Convert dict to yaml"
    yaml = YAML()
    yaml.indent(mapping=4, sequence=6, offset=3)
    yaml.dump(input_dict, sys.stdout)
  