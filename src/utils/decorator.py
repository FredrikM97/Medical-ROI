"""
Decorators for functions
"""

import functools
import os
import sys

import matplotlib.pyplot as plt

from src.files.file import write


def figure_decorator(func, figsize=(10,10)):
    """Add decoratoor to function to create subplots"

    Args:
      func: 
      figsize: (Default value = (10,10))

    Returns:

    Raises:

    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        """

        Args:
          *args: 
          **kwargs: 

        Returns:

        Raises:

        """
        fig, ax = plt.subplots(figsize=figsize)
        tmp = func(*args, fig=fig,**kwargs)
        
        plt.close(fig)
        return tmp
    return wrapper

def close_on_finish_decorator(func, filepath,*args,message='',**kwargs):
    """Call function and write to file if success. If fail then raise an error

    Args:
      func: 
      filepath: 
      *args: 
      message: (Default value = '')
      **kwargs: 

    Returns:

    Raises:

    """
    try:
        tmp = func(*args,**kwargs)
        write(filepath + "/done.sample", str(message))
        return tmp
    except Exception as e:
        raise Exception("Error occured in function") from e
    
    
class HiddenPrints:
    """Hide prints from the command line"""

    def __enter__(self):
        """ """
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        """

        Parameters
        ----------
        exc_type :
            
        exc_val :
            
        exc_tb :
            

        Returns
        -------

        
        """
        sys.stdout.close()
        sys.stdout = self._original_stdout