"""
This module contain various types of functions/classes to access and generate segmentation and bboxes from WSOL.
"""


from skimage import measure
from skimage import segmentation as seg
from skimage.filters import sobel

from scipy import ndimage as ndi
import numpy as np
import torch

import ast
from typing import Tuple, Union, List
from torchvision.ops._utils import convert_boxes_to_roi_format
import warnings

from roi_align import RoIAlign
from nms import batched_nms
from src.utils.preprocess import tensor2numpy, preprocess_image
from src.utils import plot
from src.utils.decorator import HiddenPrints

#with HiddenPrints(), warnings.catch_warnings():

def segment_mask(background_mask:np.ndarray, image_mask:np.ndarray, upper_bound=150) -> np.ndarray:
    """Simple segmentation of the mask input

    Parameters
    ----------
    background_mask : np.ndarray
        
    image_mask : np.ndarray
        
    upper_bound :
        (Default value = 150)

    Returns
    -------

    
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
    Returns:
        min_col, min_row, max_col, max_row, min_depth, max_depth
    
    Example:
    
    tmp_image = np.zeros((3,100,100)).astype(int)#np.resize(tmp_image,(3,100,100))
    tmp_image[1][30:40,20:50] = 1
    
    feature = measure.regionprops(tmp_image)[0]
    feature.bbox -> (1, 30, 20, 2, 40, 50)

    Parameters
    ----------
    feature : object
        

    Returns
    -------
    type
        (20, 30, 50, 40, 1, 2)

    
    """
    
    min_depth, min_row, min_col, max_depth, max_row, max_col = feature.bbox
    return min_col, min_row, max_col, max_row, min_depth, max_depth

def bounding_boxes(features:list):
    """Expects features from 2D images

    Parameters
    ----------
    features : list
        List

    Returns
    -------
    type
        output List[K, 6] with the box coordinates in  (x0, y0, x1, y1, z0, z1) and k is batch size.

    
    """

    if isinstance(features, list):
        return [get_bbox_coordinates(feature) for feature in features]

    else:
        return get_bbox_coordinates(features)

def roi_align(image, boxes:list, output_shape:Tuple=(40,40,40), displayed:bool=False):
    """Create aligned image rois for the neural network
    Arg:
        image: Image of shape Tuple[D,H,W]
        features (List[Tuple[int,int,int,int,int]]): List of features (z0,y0,z1,y1,x0,x1). Shape is expected based on the input of ROIAlign

    Parameters
    ----------
    image :
        
    boxes : list
        
    output_shape : Tuple
        (Default value = (40,40,40))
    displayed : bool
        (Default value = False)

    Returns
    -------

    
    """

    image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
    box_tensor = [torch.stack([torch.Tensor(x) for x in boxes]).cuda()]
    
    roialign = RoIAlign(output_shape,spatial_scale=1.0,sampling_ratio=-1)
    image_rois = roialign.forward(image_tensor,box_tensor)

    # None branched syntax
    if displayed:
        [plot.display_3D(x[0],step=1) for x in tensor2numpy(image_rois)]
    return image_rois

def column_to_tuple(pd_column):
    """Convert a pandas column from string to tuple

    Parameters
    ----------
    pd_column :
        Series

    Returns
    -------
    type
        output (Series):

    
    """
    
    return pd_column.apply(ast.literal_eval)

def column_to_np(pd_column, dtype:str='float64'):
    """Convert a pandas column from tuple to numpy arrays

    Parameters
    ----------
    pd_column :
        Series
    dtype : str
        (Default value = 'float64')

    Returns
    -------
    type
        output (Series):

    
    """
    
    return pd_column.apply(lambda x: np.array(x, dtype=dtype))

def center_coordinates(list_of_bbox:list):
    """

    Parameters
    ----------
    list_of_bbox : list
        

    Returns
    -------

    
    """
    z = (list_of_bbox[0] + list_of_bbox[2])/2
    y = (list_of_bbox[1] + list_of_bbox[3])/2
    x = (list_of_bbox[4] + list_of_bbox[5])/2
    
    return x,y,z

def max_occurance(occurances:list):
    """Calculate the number of occurances of a type

    Parameters
    ----------
    occurances : list
        

    Returns
    -------

    
    """
    u,c = np.unique(occurances, return_counts=True)
    max_val = u[c == c.max()]
    return max_val

def nms_reduction(_bboxes, th=0.5):
    """Reduce pandas dataframe data containing 'bbox', 'score' and 'observe_class'.

    Parameters
    ----------
    _bboxes :
        
    th :
        (Default value = 0.5)

    Returns
    -------

    
    """

    bbox_tensor = torch.Tensor(_bboxes['bbox'].to_list()).float()
    scores = torch.Tensor(_bboxes['score'].to_list()) 
    idxs  = torch.Tensor(_bboxes['observe_class'].to_list())

    only_interesting = bbox_tensor[batched_nms(bbox_tensor.cuda(), scores.cuda(), idxs.cuda(), th).detach().cpu()]
    return only_interesting

class Feature_extraction():
    """Extract features from CAM where the segmented image are about a given threshold.
    
    Supports two functions: features and extract.

    Parameters
    ----------

    Returns
    -------

    
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

    def extract(self, nifti_image:torch.Tensor, observe_class:int):
        """

        Parameters
        ----------
        nifti_image : torch.Tensor
            
        observe_class : int
            

        Returns
        -------
        type
            

        
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
                
    def features(self, i:int, image_name:str, nifti_image, patient_class:int, observe_class:int):
        """Calculate score and return info of the segmented features

        Parameters
        ----------
        i : int
            index
        image_name : str
            Name of image
        nifti_image :
            image from nifti
        patient_class : int
            Patient class
        observe_class : int
            Which class to observe

        Returns
        -------
        dict : 
            Features for; image, patient_class, observe_class, probability_class and score

        
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