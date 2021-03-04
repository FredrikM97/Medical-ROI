import os
import matplotlib.pyplot as plt
import numpy as np

__all__ = ['get_availible_files','to_cpu_numpy','merge_dict','normalize','batchisize_to_5D','move_to_device']

def get_availible_files(path, contains:str=''):
    return [f for f in os.listdir(path) if contains in f]
    
def to_cpu_numpy(data):
    # Send to CPU. If computational graph is connected then detach it as well.
    if data.requires_grad:
        print("Disconnect graph!")
        return data.detach().cpu().numpy()
    else:
        return data.cpu().numpy()
            
def merge_dict(a, b, path=None):
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

normalize = lambda x: (x - x.min())/(x.max()-x.min())
batchisize_to_5D = lambda x: x.expand((*[1]*(5-len(x.shape)),*[-1]*len(x.shape)))
move_to_device = lambda model, image, device:  (model.to(device), image.to(device))
