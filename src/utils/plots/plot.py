import matplotlib.pyplot as plt
import functools
from skimage.transform import resize
import numpy as np
import torchvision
import torch
from src.utils.preprocess import normalize, image2axial, greedy_split
from torch import Tensor
from typing import Tuple
from abc import abstractmethod


def figure_decorator(func, figsize=(10,10)):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        fig, ax = plt.subplots(figsize=figsize)
        tmp = func(*args, fig=fig,**kwargs)
        
        plt.close(fig)
        return tmp
    return wrapper

class Plot:
    """
        Custom plot function to simplify the generation of images and preprocess
        
    """
    @abstractmethod
    def plot(self, fig=None):
        pass
        #plt.imshow(image)
                     
    def _resize(self, image:np.ndarray,input_shape:Tuple=None):
        """Expect batch to be included in shape here"""
        image = resize(image,input_shape)
        assert image.shape == input_shape, f"Expected shape; {input_shape}, Got; {image.shape}"
        return image
    
    def uint8(self,image:np.ndarray):
        # Set the pixel boundary between 0-255 instead of 0-1 and change type to uint8
        return (normalize(image) * 255).astype(np.uint8)
    
    def preprocess(self, image:np.ndarray, input_shape:Tuple=(79,95,79)) -> Tensor:
        """Resize, normalize between 0-1"""
        
        isinstance(image, np.ndarray)
        image = self._resize(image,input_shape)
        image = normalize(image)
        image = self.uint8(image)
        
        return image
    
    def grid(self, image:np.ndarray, max_num_slices=16,pad_value=0.5) -> Tensor:
        """ Create grid from image based on maximum number of slices.
        
        Args:
            * image: Image with multiple slices of shape Tuple[D,H,W]
            * max_num_slices: The number of slices the image should be reduced.
            
        Return:
            * Return a grid image with reduced number of slices.
        
        """

        assert image.shape == self.input_shape, f"Wrong shape, Image: {image.shape}, Expected: {self.input_shape}"
        nrow=9
        if max_num_slices != None:
            image = np.stack([np.mean(x,axis=0) for x in greedy_split(image,max_num_slices)])
            nrow=4
        
        plt_image = torch.from_numpy(image).float().unsqueeze(1)
        
        # Convert to grid 
        grid_image = torchvision.utils.make_grid(plt_image, nrow=nrow,pad_value=pad_value, normalize=True)[0]*255
        return grid_image