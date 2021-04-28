import matplotlib.pyplot as plt
import functools

def figure_decorator(func, figsize=(10,10)):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        fig, ax = plt.subplots(figsize=figsize)
        tmp = func(*args, fig=fig,**kwargs)
        
        plt.close(fig)
        return tmp
    return wrapper

def close_on_finish_decorator(func, filepath,*args,**kwargs):
    try:
        tmp = func(*args,**kwargs)
        open(filepath + "/done.sample", 'a').close()
        return tmp
    except Exception as e:
        raise Exception("Error occured in function") from e
    