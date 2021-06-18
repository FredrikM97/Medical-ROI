"""
This module contain various types of functions/classes to access and generate CAM.

"""
from src.files import preprocess
from src.files.preprocess import image2axial, to_grid
import nibabel as nib

import warnings
import torchcam 
import torch
#from torch import tensor
import numpy as np
from typing import Tuple, Union, List

import torchvision
import matplotlib.pyplot as plt

from src.display.cmap import parula_map


class CAM:
    """ """
    def __init__(self, model, cam_type=torchcam.cams.GradCAMpp,target_layer:str="model.layer4", cam_kwargs:dict={}):
        """

        Parameters
        ----------
        model :
            
        cam_type :
            (Default value = torchcam.cams.GradCAMpp)
        target_layer : str
            (Default value = "model.layer4")
        cam_kwargs : dict
            (Default value = {})

        Returns
        -------

        
        """
        self._CLASSES=[0,1,2]
        self.CAM_TYPE = cam_type
        self.TARGET_LAYER = target_layer
        self.model = model
        self.extractor = cam_type(model, target_layer=target_layer, **cam_kwargs)

    def class_score(self, input_image:np.ndarray, device='cuda', input_shape=(79,95,79)) -> Tuple[torch.tensor, int]:
        """Calculate the class scores and the highest probability of the target class

        Parameters
        ----------
        input_image : np.ndarray
            
        device :
            (Default value = 'cuda')
        input_shape :
            (Default value = (79,95,79))

        Returns
        -------
        Tuple[torch.tensor,int]
            All the probabilities and the best probability class

        
        """
        image = preprocess.preprocess_image(input_image)
        image = preprocess.batchisize_to_5D(image)
        image_tensor = torch.from_numpy(image).float()
        
        model = self.model.to(device).eval()
        image_tensor = image_tensor.to(device)
        # Check that image have the correct shape
        assert tuple(image_tensor.shape) == (1, 1, *input_shape), f"Got image shape: {image_tensor.shape} expected: {(1, 1, *input_shape)}"
        assert model.device == image_tensor.device, f"Model and image are not on same device: Model: {model.device} Image: {image_tensor.device}"
  
        class_scores = model(image_tensor)
        return class_scores, class_scores.squeeze(0).argmax().item()
    
    def activations(self, class_idx:int=None, class_scores:torch.tensor=None) -> np.ndarray:
        """Retrieve the map based on the score from the model

        Parameters
        ----------
        class_idx : int
            (Default value = None)
        class_scores : torch.tensor
            (Default value = None)

        Returns
        -------
        np.ndarray
            Tensor with activations from image with shape tensor[D,H,W]

        
        """

        return self.extractor(class_idx, class_scores, normalized=False).detach().cpu()
    
    @staticmethod
    def plot(images:list=[], masks:list=[], labels=[],cmap=parula_map, alpha:float=0.7, class_label:str=None, predicted_override:bool=None, architecture:str=None):
        """Create a plot from the given class activation map and input image. CAM is calculated from the models weights and the probability distribution of each class.

        Parameters
        ----------
        images : list
            (Default value = [])
        masks : list
            (Default value = [])
        labels :
            (Default value = [])
        cmap :
            Color object (Default value = parula_map)
        alpha : float
            int (Default value = 0.7)
        class_label : str
            str (Default value = None)
        predicted_override : bool
            bool (Default value = None)
        architecture : str
            (Default value = None)

        Returns
        -------
        type
            output (Figure): Figure reference to plot

        
        """
        #class_idx = class_idx if isinstance(class_idx, list) else [class_idx]
        if (max_length :=len(masks)) > len(images):
            pass
        else:
            max_length = len(images)
        
        if max_length == 0:
            raise ValueError("Number of images/masks cant be zero!")
        
        fig, axes = plt.subplots(ncols=max_length,nrows=1,figsize=(max_length*8,8))
        
        if max_length > 1:
            # Add images
            for i, image in enumerate(images):
                im = axes[i].imshow(image,cmap='Greys_r', vmin=image.min(), vmax=image.max())


            # Add masks
            for i, mask in enumerate(masks):
                im = axes[i].imshow(mask,cmap=cmap, alpha=alpha,vmin=mask.min(), vmax=mask.max()) 
        
        else:
            for i, image in enumerate(images):
                im = axes.imshow(image,cmap='Greys_r', vmin=image.min(), vmax=image.max())


            # Add masks
            for i, mask in enumerate(masks):
                im = axes.imshow(mask,cmap=cmap, alpha=alpha,vmin=mask.min(), vmax=mask.max()) 
                
        # Add labels
        classes = {
            0:'CN',
            1:'MCI',
            2:'AD'
        }
        
        for i, label in enumerate(labels):
            title_list = [out for out, con in [
                (f'{architecture}',architecture),
                #(f'{type(self.extractor).__name__}',True),
                (f'Patient: {class_label}',class_label),
                (f'Predicted: {classes[label]}',label),
                (f'Overrided',predicted_override)] if con != None
            ]
            if max_length > 1:
                axes[i].set_title(', '.join(title_list))

            else:
                axes.set_title(', '.join(title_list))
        
        if max_length > 1:
            for a in axes.flatten():
                a.set_axis_off()
                a.set_xticklabels([])
                a.set_yticklabels([])
        else:
            axes.set_axis_off()
            axes.set_xticklabels([])
            axes.set_yticklabels([])
            
        # Remove axis data to show colorbar more clean
        ax = axes.ravel().tolist() if max_length > 1 else axes
        plt.subplots_adjust(wspace=0.01, hspace=0)
        cbar = fig.colorbar(im, ax=ax, shrink=1)
 
        return fig

    @staticmethod
    def get_cam(model, cam_type, input_shape:Tuple=(79,95,79),target_layer:str=None,CAM_kwargs:dict={}):
        """Generate CAM object

        Parameters
        ----------
        model :
            
        cam_type :
            
        input_shape : Tuple
            (Default value = (79,95,79))
        target_layer : str
            (Default value = None)
        CAM_kwargs : dict
            (Default value = {})

        Returns
        -------

        
        """
        extractor = cam_type(model, input_shape=(1,*input_shape), target_layer=target_layer, **CAM_kwargs)
        return extractor
    
    @staticmethod
    def average_image(images:list):
        """Calculate average over multiple images

        Parameters
        ----------
        images : list
            

        Returns
        -------

        
        """
        return torch.mean(torch.stack(images), axis=0)
    
    @staticmethod
    def repeat_stack(image:torch.tensor, repeat:int=1, grid_kwargs:dict={}):
        """Repeat am image in a grid N number of times.

        Parameters
        ----------
        image : torch.tensor
            
        repeat : int
            (Default value = 1)
        grid_kwargs : dict
            (Default value = {})

        Returns
        -------

        
        """
        return torch.stack([to_grid(image, **grid_kwargs)]*repeat)
    
    @staticmethod
    def preprocess(filename:str):
        """Preprocess image to a valid format

        Parameters
        ----------
        filename : str
            

        Returns
        -------

        
        """
        class_label = utils.split_custom_filename(filename,'/')[4]
        image = image2axial(nib.load(filename).get_fdata())
        image[image <= 0]=0
        image = preprocess.preprocess_image(image)
        return image