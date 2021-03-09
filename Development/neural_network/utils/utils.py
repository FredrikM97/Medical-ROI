import os
import matplotlib.pyplot as plt
import numpy as np
import functools
from skimage.transform import resize
import torch
import torchvision
import nibabel as nib
from typing import Tuple

__all__ = [
    'get_availible_files','to_cpu_numpy','merge_dict',
    'normalize','batchisize_to_5D','move_to_device', 'figure_decorator',
    'to_np_fig_grid','image2axial','load_nifti','mask_threshold',
    'set_outside_of_scan_pixels_2_zero','greedy_split'
]

def get_availible_files(path, contains:str=''):
    return [f for f in os.listdir(path) if contains in f]
    
def to_cpu_numpy(data):
    # Send to CPU. If computational graph is connected then detach it as well.
    if data.requires_grad:
        print("Disconnect graph!")
        return data.detach().cpu().numpy()
    else:
        return data.cpu().numpy()
            
def merge_dict(a:dict, b:dict, path=None):
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

def normalize(image:np.ndarray) -> np.ndarray:
    # Normalize an image with min-max-normalization
    
    return (image - image.min())/(image.max()-image.min())

def batchisize_to_5D(image:np.ndarray) -> np.ndarray:
    # Convert a numpy array to 5D if its length is less than 5
    return image.expand((*[1]*(5-len(image.shape)),*[-1]*len(image.shape)))

def move_to_device(model, image, device):
    # Move an image and model to a device
    return (model.to(device), image.to(device))


def figure_decorator(func, figsize=(20,20)):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        fig, ax = plt.subplots(figsize=figsize)
        tmp = func(*args, fig=fig,**kwargs)
        
        #plt.close(fig)
        return tmp
    return wrapper

def to_np_fig_grid(image:np.ndarray, resize_shape:Tuple[int,int,int]=(79,95,79)):
    # Convert an images to a torch grid
    isinstance(image, np.ndarray)
    image = resize(image,resize_shape)
    image = torch.from_numpy((image * 255).astype(np.uint8)).unsqueeze(1)
    
    assert image.shape == (resize_shape[0], 1, *resize_shape[1:])
    
    grid_img = torchvision.utils.make_grid(image, nrow=10)[0]
    
    return grid_img

def to_np_figure(image:np.ndarray, resize_shape:Tuple[int,int,int]=(79,224,224)):
    isinstance(image, np.ndarray)
    image = resize(image,resize_shape)
    image = torch.from_numpy((image * 255).astype(np.uint8)).unsqueeze(1)
    
    assert image.shape == (resize_shape[0], 1, *resize_shape[1:])
    
    return image

def image2axial(image:np.ndarray) -> np.ndarray:
    # Modify image to display axial view. Expects an image of (B,H,W) or (B,C,H,W) 
    if not isinstance(image, np.ndarray): raise TypeError(f"Expected np.ndarray. Got: {type(image)}")
    if len(image.shape) == 3:
        return image.transpose(2,1,0)
    elif len(image.shape) == 4:
        return image.transpose(3,1,2,0)
    else:
        raise ValueError(f"Expected length of 3 or 4. Got: {len(image.shape)}")
        
        
def mask_threshold(image, threshold:float):
    # Based on a threshold value. Set everything below threshold to zero.
    img_threshold = image.max()*threshold
    return np.ma.masked_where(image <= img_threshold,image)


def set_outside_of_scan_pixels_2_zero(image:np.ndarray, mask:np.ndarray) -> np.ndarray:
    # Everything that is zero outside of image is set to zero on the mask. Returns the modified mask
    return np.ma.masked_where(image == 0,mask)

def load_nifti(path:str) -> np.ndarray:
    # Load nifti image and convert to axial view
    return image2axial(nib.load(path).get_fdata())

def greedy_split(arr, n, axis=0) -> list:
    length = arr.shape[axis]

    # compute the size of each of the first n-1 blocks
    block_size = np.ceil(length / float(n))

    # the indices at which the splits will occur
    ix = np.arange(block_size, length, block_size).astype(np.uint8)
    return np.split(arr, ix, axis)