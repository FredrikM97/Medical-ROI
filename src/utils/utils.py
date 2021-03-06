import os
import matplotlib.pyplot as plt
import numpy as np
import functools
from skimage.transform import resize
import torch
import torchvision

__all__ = ['get_availible_files','to_cpu_numpy','merge_dict','normalize','batchisize_to_5D','move_to_device', 'figure_decorator','to_np_fig_grid']

def availible_files(path, contains:str=''):
    return [f for f in os.listdir(path) if contains in f]
    
def tensor2numpy(data):
    # Send to CPU. If computational graph is connected then detach it as well.
    if data.requires_grad:
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

def normalize(x):
    return (x - x.min())/(x.max()-x.min())

def batchisize_to_5D(x):
    return x.expand((*[1]*(5-len(x.shape)),*[-1]*len(x.shape)))

def move_to_device(model, image, device):
    return (model.to(device), image.to(device))

def figure_decorator(func, figsize=(20,20)):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        fig, ax = plt.subplots(figsize=figsize)
        tmp = func(*args, fig=fig,**kwargs)
        
        plt.close(fig)
        return tmp
    return wrapper

def numpy2grid(image, resize_shape=(79,224,224)):
    isinstance(image, np.ndarray)
    image = resize(image,resize_shape)
    image = torch.from_numpy((image * 255).astype(np.uint8)).unsqueeze(1)
    
    assert image.shape == (resize_shape[0], 1, *resize_shape[1:])
    
    grid_img = torchvision.utils.make_grid(image, nrow=10)[0]
    
    return grid_img

def numpy2figure(image, resize_shape=(79,224,224)):
    isinstance(image, np.ndarray)
    image = resize(image,resize_shape)
    image = torch.from_numpy((image * 255).astype(np.uint8)).unsqueeze(1)
    
    assert image.shape == (resize_shape[0], 1, *resize_shape[1:])
    
    return image