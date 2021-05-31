import matplotlib.pyplot as plt
import functools
import os, sys
from .print import write_to_file

def figure_decorator(func, figsize=(10,10)):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        fig, ax = plt.subplots(figsize=figsize)
        tmp = func(*args, fig=fig,**kwargs)
        
        plt.close(fig)
        return tmp
    return wrapper

def close_on_finish_decorator(func, filepath,*args,message='',**kwargs):
    try:
        tmp = func(*args,**kwargs)
        write_to_file(filepath + "/done.sample", str(message))
        return tmp
    except Exception as e:
        raise Exception("Error occured in function") from e
    
    
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout