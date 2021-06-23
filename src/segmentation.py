"""
This module contain various types of functions/classes to access and generate segmentation and bboxes from WSOL.
"""


import ast
import warnings
from typing import List, Tuple, Union

import numpy as np
import torch
from nms import batched_nms
from scipy import ndimage as ndi
from skimage import measure
from skimage import segmentation as seg

from src.display import plot
from src.files.preprocess import preprocess_image, tensor2numpy
from src.utils.decorator import HiddenPrints


def segment_mask(background_mask:np.ndarray, image_mask:np.ndarray, upper_bound=150) -> np.ndarray:
    """Simple segmentation of the mask input

    Args:
      background_mask(np.ndarray): 
      image_mask(np.ndarray): 
      upper_bound: (Default value = 150)

    Returns:

    Raises:

    """

    tmp_image = image_mask.copy()
    tmp_image[tmp_image < upper_bound] = 0
    tmp_image[background_mask<=0] = 0
    tmp_image[tmp_image>0] = 1
    labeled_masks = measure.label(tmp_image, background=0)#ndi.label(tmp_image)[0]

    return labeled_masks

def get_bbox_coordinates(feature:object):
    """Convert skimage format to x,y,x,y,z0,z1
    
    Expects feature.bbox to be in format:
        min_depth, min_row, min_col, max_depth, max_row, max_col

    Args:
      feature(object): 

    Returns:
      : min_col, min_row, max_col, max_row, min_depth, max_depth
      Example: 
      Example: tmp_image = np.zeros((3,100,100)).astype(int)#np.resize(tmp_image,(3,100,100))
      tmp_image[1][30: 40,20:50] = 1
      tmp_image[1][30: 40,20:50] = 1
      tmp_image[1][30: 40,20:50] = 1
      tmp_image[1][30: 40,20:50] = 1
      tmp_image[1][30: 40,20:50] = 1
      feature = measure.regionprops(tmp_image)[0]
      tmp_image[1][30: 40,20:50] = 1
      tmp_image[1][30: 40,20:50] = 1
      tmp_image[1][30: 40,20:50] = 1
      tmp_image[1][30: 40,20:50] = 1
      feature = measure.regionprops(tmp_image)[0]
      tmp_image[1][30: 40,20:50] = 1
      tmp_image[1][30: 40,20:50] = 1
      tmp_image[1][30: 40,20:50] = 1
      feature = measure.regionprops(tmp_image)[0]
      tmp_image[1][30: 40,20:50] = 1
      tmp_image[1][30: 40,20:50] = 1
      feature = measure.regionprops(tmp_image)[0]
      tmp_image[1][30: 40,20:50] = 1
      feature = measure.regionprops(tmp_image)[0]
      feature.bbox -> (1, 30, 20, 2, 40, 50)

    Raises:

    """
    
    min_depth, min_row, min_col, max_depth, max_row, max_col = feature.bbox
    return min_col, min_row, max_col, max_row, min_depth, max_depth

def bounding_boxes(features:list):
    """Expects features from 2D images

    Args:
      features(list): List

    Returns:
      type: output List[K, 6] with the box coordinates in  (x0, y0, x1, y1, z0, z1) and k is batch size.

    Raises:

    """

    if isinstance(features, list):
        return [get_bbox_coordinates(feature) for feature in features]

    else:
        return get_bbox_coordinates(features)

def center_coordinates(list_of_bbox:list) -> 'Tuple[float,float,float]':
    """

    Args:
      list_of_bbox(list): 

    Returns:

    Raises:

    """
    z = (list_of_bbox[0] + list_of_bbox[2])/2
    y = (list_of_bbox[1] + list_of_bbox[3])/2
    x = (list_of_bbox[4] + list_of_bbox[5])/2
    
    return x,y,z

def max_occurance(occurances:list) ->'np.ndarray':
    """Calculate the number of occurances of a type

    Args:
      occurances(list): 

    Returns:

    Raises:

    """
    u,c = np.unique(occurances, return_counts=True)
    max_val = u[c == c.max()]
    return max_val

def nms_reduction(_bboxes, th=0.5) -> list:
    """Reduce pandas dataframe data containing 'bbox', 'score' and 'observe_class'.

    Args:
      _bboxes: 
      th: (Default value = 0.5)

    Returns:

    Raises:

    """

    bbox_tensor = torch.Tensor(_bboxes['bbox'].to_list()).float()
    scores = torch.Tensor(_bboxes['score'].to_list()) 
    idxs  = torch.Tensor(_bboxes['observe_class'].to_list())

    only_interesting = bbox_tensor[batched_nms(bbox_tensor.cuda(), scores.cuda(), idxs.cuda(), th).detach().cpu()]
    return only_interesting

class Feature_extraction():
    """Extract features from CAM where the segmented image are about a given threshold.
    
    Supports two functions: features and extract.

    Args:

    Returns:

    Raises:

    """
    def __init__(self, cam_extractor, upper_bound:float=0.85, use_quantile_bounds:bool=True,lambda1:int=1,lambda2:int=1, n_average:int=2):
        """

        Parameters
        ----------
        cam_extractor :
            
        upper_bound : float
            (Default value = 0.85)
        use_quantile_bounds : bool
            (Default value = True)
        lambda1 : int
            (Default value = 1)
        lambda2 : int
            (Default value = 1)
        n_average : int
            (Default value = 2)

        Returns
        -------

        
        """

        self.cam_extractor = cam_extractor
        self.upper_bound = upper_bound
        self.use_quantile_bounds = use_quantile_bounds
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.n_average = n_average

    def extract(self, nifti_image:'torch.Tensor', observe_class:int) -> tuple:
        """Extract features from the medical images over an average of N samples to create a segmentation mask which are then

        Args:
          nifti_image('torch.Tensor'): 
          observe_class(int): 

        Returns:
          : Tuple containing a segmented mask, the image mask (CAM) which class index that is observed and the segmentation threshold

        Raises:

        """
        if len(nifti_image.shape) > 3:
            nifti_image = nifti_image.squeeze(0)

        np_image = tensor2numpy(nifti_image)
        masks = []
        
        for x in range(self.n_average):
            cam_extractor = self.cam_extractor()
            class_scores, class_idx = cam_extractor.class_score(nifti_image) #evaluate
            masks.append(cam_extractor.activations(observe_class, class_scores))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            image_mask = preprocess_image(torch.mean(torch.stack(masks), axis=0))
        
        image_mask = (image_mask*255).astype(int)
        np_image = (np_image*255).astype(int)
        
        # If quantiles is enable then calculate it and use it instead based on the upper bound
        if self.use_quantile_bounds:
            _upper_bound = np.quantile(image_mask.ravel(), self.upper_bound)
        else:
            _upper_bound = self.upper_bound
            
        image_mask[np_image == 0] = 0
        
        segmented_mask = segment_mask(np_image, image_mask, upper_bound=_upper_bound)
        
        return segmented_mask, image_mask,class_idx,_upper_bound
                
    def features(self, i:int, image_name:str, nifti_image, patient_class:int, observe_class:int) -> dict:
        """Calculate score and return info of the segmented features

        Args:
          i(int): index
          image_name(str): Name of image
          nifti_image: image from nifti
          patient_class(int): Patient class
          observe_class(int): Which class to observe

        Returns:
          : A dictionary features containing the info; image name, patient class, observe class, probability class and score

        Raises:

        """
        segmented_mask, image_mask, class_idx,_upper_bound = self.extract(nifti_image, observe_class)
        features = measure.regionprops(segmented_mask, intensity_image=image_mask)
        new_features = {
            'bbox_area':[feature.bbox_area for feature in features], 
            'mean_intensity':[feature.mean_intensity for feature in features], 
            'bbox':bounding_boxes(features),
            'upper_bound':_upper_bound,
            'use_quantile_bounds':self.use_quantile_bounds,
        }
        new_features.update({

            'score':self.lambda1 * np.mean(new_features['mean_intensity'])/np.max(image_mask) - self.lambda2*((np.array(new_features['bbox_area']))/image_mask.size)
        })
        

        return {'image':image_name, 'patient_class':patient_class, 'observe_class':observe_class, 'probability_class':class_idx, **new_features}