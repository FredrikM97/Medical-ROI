from src.utils import utils, preprocess
#from src.utils.decorator import HiddenPrints

import nibabel as nib
from src.utils.preprocess import image2axial, to_grid
import warnings
import torchcam 
import torch
from torch import tensor
import numpy as np
from typing import Tuple, Union, List
import enum

import torchvision
import matplotlib.pyplot as plt

from src.utils.cmap import parula_map
#from src.classifier.agent import Agent

class CAM:
    def __init__(self, model, cam_type=torchcam.cams.GradCAMpp,target_layer="model.layer4", cam_kwargs={}):
        self._CLASSES=[0,1,2]
        self.CAM_TYPE = cam_type
        self.TARGET_LAYER = target_layer
        self.model = model
        self.extractor = cam_type(model, target_layer=target_layer, **cam_kwargs)

    def class_score(self, input_image:np.ndarray, device='cuda', input_shape=(79,95,79)) -> Tuple[tensor, int]:
        """ Calculate the class scores and the highest probability of the target class
            
        Args:
        
        Return:
            * Tuple containing all the probabilities and the best probability class
        """
        image = preprocess.preprocess_image(input_image)
        image = preprocess.batchisize_to_5D(image)
        image_tensor = torch.from_numpy(image).float()
        
        self.model.to(device)
        image_tensor = image_tensor.to(device)
        # Check that image have the correct shape
        assert tuple(image_tensor.shape) == (1, 1, *input_shape), f"Got image shape: {image_tensor.shape} expected: {(1, 1, *input_shape)}"
        assert self.model.device == image_tensor.device, f"Model and image are not on same device: Model: {model.device} Image: {image_tensor.device}"
  
        self.model.eval()
        class_scores = self.model(image_tensor)
        return class_scores, class_scores.squeeze(0).argmax().item()
    '''
    @staticmethod
    def activations2grid(class_scores:tensor, class_idx:Union[List[int],int], input_image:np.ndarray, normalized=True, grid_kwargs={},**kwargs) -> Tuple[tensor, tensor]:
        """Creates a grid based on a class_idx."""
        
        
        if isinstance(class_idx, list) and len(class_idx) == 1:
            class_idx = class_idx[0]

        if isinstance(class_idx, list):
            grid_img = torch.hstack([
                preprocess.to_grid(preprocess.normalize(input_image), **grid_kwargs.copy())
            ]*len(class_idx))
            
            grid_mask = torch.hstack([
                preprocess.to_grid(preprocess.normalize(self.activations(extractor, idx, class_scores)), **grid_kwargs.copy())
                for idx in class_idx
            ])
        
            
        elif isinstance(class_idx, int):
            grid_mask = preprocess.to_grid(preprocess.normalize(self.activations(extractor, class_idx, class_scores)), **grid_kwargs.copy())
            grid_img = preprocess.to_grid(preprocess.normalize(input_image), **grid_kwargs.copy())
            
        else:
            raise ValueError(f"Expected class_idx of type list or int, Got: {type(class_idx)}")
            
        return grid_img, grid_mask
    '''
    
    def activations(self, class_idx:int=None, class_scores:tensor=None) -> np.ndarray:
        """ Retrieve the map based on the score from the model
        
        Return:
            * Tensor with activations from image with shape Tensor[D,H,W]
        """

        return self.extractor(class_idx, class_scores, normalized=False).detach().cpu()
    
    @staticmethod
    def plot(images=[], masks=[], labels=[],cmap=parula_map, alpha=0.7, class_label:str=None, predicted_override=None, architecture=None):
        """Create a plot from the given class activation map and input image. CAM is calculated from the models weights and the probability distribution of each class.
        
        Args:
            class_scores (Tensor): Tensor of probability for each class 
            class_idx (Union[List[int],int]): Index of highest class_score probability 
            input_image (np.ndarray): image to the evaluated
            cmap (Color object), optional: The cmap to use in plot 
            alpha (int), optional: Alpha value for opacity 
            class_label (str), optional: Which class the image belongs to.
            predicted_override (bool), optional: If class_idx should be overwritten and plot each class instead
            max_num_slices (int), optional: Number of total slices that should be plotted. Combine multiple slices if image slices are more than number of max_num_slices.
            
        Return:
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
    def get_cam(model, cam_type, input_shape=(79,95,79),target_layer=None,CAM_kwargs={}):
        #with HiddenPrints(), warnings.catch_warnings():
        #    warnings.simplefilter("ignore")
        extractor = cam_type(model, input_shape=(1,*input_shape), target_layer=target_layer, **CAM_kwargs)
        return extractor
    
    @staticmethod
    def average_image(images):
        return torch.mean(torch.stack(images), axis=0)
    
    @staticmethod
    def repeat_stack(image, repeat=1, grid_kwargs={}):
        return torch.stack([to_grid(image, **grid_kwargs)]*repeat)
    
    @staticmethod
    def preprocess(filename):
        class_label = utils.split_custom_filename(filename,'/')[4]
        image = image2axial(nib.load(filename).get_fdata())
        image[image <= 0]=0
        image = preprocess.preprocess_image(image)
        return image


def new_seed():
    return torch.manual_seed(torch.seed())

"""
def get_agent(checkpoint_path):
    model = None
    with HiddenPrints() as f:#, warnings.catch_warnings():
        trainer = Agent(checkpoint_path=checkpoint_path)
        trainer.load_model()
        model = trainer.model
    return model
    
"""