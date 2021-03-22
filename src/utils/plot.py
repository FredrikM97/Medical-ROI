import matplotlib.pyplot as plt
import functools
from skimage.transform import resize
import numpy as np
import torchvision
import torch

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
