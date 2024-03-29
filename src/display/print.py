"""
Common print functions used in different modules.
"""



import sys
from math import ceil, floor

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from ruamel.yaml import YAML


def pd_cols(input_df:pd.DataFrame) -> None:
    """Plot all columns of a pandas dataframe

    Args:
      input_df(pd.DataFrame): 

    Returns:

    Raises:

    """
    "Print all pandas columns"
    with pd.option_context('display.max_columns', None):
        display(input_df.head())

def dict2yaml(input_dict:dict) -> None:
    """Plot a dictionary in yaml format.

    Args:
      input_dict(dict): 

    Returns:

    Raises:

    """
    "Convert dict to yaml"
    yaml = YAML()
    yaml.indent(mapping=4, sequence=6, offset=3)
    yaml.dump(input_dict, sys.stdout)
  