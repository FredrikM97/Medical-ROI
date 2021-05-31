from skimage import measure
from skimage import segmentation as seg
from skimage.filters import sobel

from scipy import ndimage as ndi
import numpy as np
import torch

import ast
from typing import Tuple, Union, List
from torchvision.ops._utils import convert_boxes_to_roi_format

from src.dependencies.roi_align import RoIAlign
from src.dependencies.nms import batched_nms
from src.utils.preprocess import tensor2numpy, preprocess_image
from src.utils import plot

def remove_known_background_from_mask(image:np.ndarray, image_mask:np.ndarray) -> np.ndarray:
    """We know that background cant be useful activations. Remove them"""
    im = image_mask.copy() 
    im[image == 0] = 0
    return im

def check_join_segmentations(mask_one:np.ndarray, mask_two:np.ndarray) -> np.ndarray:
    """Observe the intersection of two masks"""
    from skimage.segmentation import join_segmentations
    return join_segmentations(mask_one, mask_two)

def segment_mask(background_mask:np.ndarray, image_mask:np.ndarray, upper_bound=150) -> np.ndarray:
    """Simple segmentation of the mask input
    Note: This one might need some scientific rework since it is not very good
    """
    """
    n=1
    im = filters.gaussian(image_mask, sigma=1/(4.*n), mode='nearest')
    mask = im > im.max()*0.62
    label_im = measure.label(mask, connectivity=None) # Use all three dimensions to decide labels
    #label_im, nb_labels = ndimage.label(mask, connectivity=1)
    return label_im
    """
    tmp_image = image_mask.copy()
    tmp_image[tmp_image < upper_bound] = 0
    tmp_image[background_mask<=0] = 0
    tmp_image[tmp_image>0] = 1
    labeled_masks = measure.label(tmp_image, background=0)#ndi.label(tmp_image)[0]
    #elevation_map = sobel(image_mask)
    
    # Set to zero so it does not break background!
    #markers = np.zeros_like(image_mask)
    
    # Get max value
    # Remove known background or values outside of thesholds
    #markers[image_mask < lower_bound] = 1
    #markers[background_mask == 0] = 1
    #markers[image_mask > upper_bound] = 2
    #segmentation_masks = seg.watershed(elevation_map, markers=markers)
    
    #segmentation_masks_new = ndi.binary_fill_holes(segmentation_masks - 1)

    #labeled_masks, _ = ndi.label(segmentation_masks_new)
    #labeled_masks = measure.label(segmentation_masks_new)
    #segmentation.display(labeled_coins, step=1)
    return labeled_masks

def get_bbox_coordinates(feature):
    """Convert skimage format to x,y,x,y,z0,z1"""
    z0, y0, x0, z1, y1, x1 = feature.bbox
    return x0,y0,x1,y1,z0,z1

def bounding_boxes(features:list):
    """Expects features from 2D images
    
    Args:
        features (List[RegionProperties])
    Return:
        output List[K, 6] with the box coordinates in  (x0, y0, x1, y1, z0, z1) and k is batch size.
    """

    if isinstance(features, list):
        return [get_bbox_coordinates(feature) for feature in features]

    else:
        return get_bbox_coordinates(features)

def roi_align(image, boxes:list, output_shape=(40,40,40), displayed=False):
    """ Create aligned image rois for the neural network
    Arg:
        image: Image of shape Tuple[D,H,W]
        features (List[Tuple[int,int,int,int,int]]): List of features (z0,y0,z1,y1,x0,x1). Shape is expected based on the input of ROIAlign
    """

    image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
    box_tensor = [torch.stack([torch.Tensor(x) for x in boxes]).cuda()]
    
    roialign = RoIAlign(output_shape,spatial_scale=1.0,sampling_ratio=-1)
    image_rois = roialign.forward(image_tensor,box_tensor)

    # None branched syntax
    if displayed:
        [plot.display_3D(x[0],step=1) for x in tensor2numpy(image_rois)]
    return image_rois

def sequential_processing(image:np.ndarray, image_mask:np.ndarray) -> None:
    """Run each processing of mask to segmentation"""
    
    # Remove mask that we know are a background (not a part of the brain scan)
    #mask_no_background = remove_known_background_from_mask(image, image_mask)
    
    # Rescale mask and segment it
    #mask_mean = mask_logarithmic_scale(mask_no_background)
    segmented_mask = segment_mask(image,image_mask)
    # Extract features with the intensities of the mask with removed background and plot it
    features =  measure.regionprops(segmented_mask, intensity_image=image_mask) #extract_features(segmented_mask, mask_no_background)
    plot.features_regions(bounding_boxes(features), image_mask)
    
    return features

def column_to_tuple(pd_column):
    """Convert a pandas column from string to tuple
    
    Args:
        pd_column (Series): A selected column to convert the content to tuple type.
    Return:
        output (Series):
    
    """
    
    return pd_column.apply(ast.literal_eval)

def column_to_np(pd_column, dtype='float64'):
    """Convert a pandas column from tuple to numpy arrays
    
    Args:
        pd_column (Series): A selected column to convert the content to numpy.
    Return:
        output (Series):
    """
    
    return pd_column.apply(lambda x: np.array(x, dtype=dtype))

def center_coordinates(list_of_bbox):
    z = (list_of_bbox[0] + list_of_bbox[2])/2
    y = (list_of_bbox[1] + list_of_bbox[3])/2
    x = (list_of_bbox[4] + list_of_bbox[5])/2
    
    return x,y,z

def max_occurance(occurances:list):
    u,c = np.unique(occurances, return_counts=True)
    max_val = u[c == c.max()]
    return max_val

def nms_reduction(_bboxes, image_mask=None, th=0.5):
    bbox_tensor = torch.Tensor(_bboxes['bbox'].to_list()).float()
    scores = torch.Tensor(_bboxes['score'].to_list()) 
    idxs  = torch.Tensor(_bboxes['observe_class'].to_list())

    only_interesting = bbox_tensor[batched_nms(bbox_tensor.cuda(), scores.cuda(), idxs.cuda(), th).detach().cpu()]
    if isinstance(image_mask,type(None)):
        image_mask = np.zeros((79,95,79))
        
    #image = add_image_bboxes(np.zeros((79,95,79)),only_interesting.numpy().astype(int))
    #fig = plot.features_regions(only_interesting.numpy().astype(int),image_mask)
    #fig = display(image, step=1)
    return only_interesting#, fig

class Feature_extraction():
    """
    Extract features from CAM where the segmented image are about a given threshold.
    
    Supports two functions: features and extract.
    """
    def __init__(self, cam_extractor, upper_bound=0.85, use_quantile_bounds=True,lambda1=1,lambda2=1, min_area=100, n_average=2):
        self.cam_extractor = cam_extractor
        self.upper_bound = upper_bound
        self.use_quantile_bounds = use_quantile_bounds
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.min_area = min_area
        self.n_average = n_average

    def extract(self, nifti_image:torch.Tensor, observe_class:int):
        """
        Return the segmented mask, image mask class index and upper_bound
        """
        if len(nifti_image.shape) > 3:
            nifti_image = nifti_image.squeeze(0)

        np_image = tensor2numpy(nifti_image)
        #print("Range of nifti images",nifti_image.max(), nifti_image.min())
        masks = []
        for x in range(self.n_average):
            class_scores, class_idx = self.cam_extractor.evaluate(nifti_image)
            masks.append(self.cam_extractor.activation_map(observe_class, class_scores))
        

        image_mask = preprocess_image(torch.mean(torch.stack(masks), axis=0))
        
        image_mask = (image_mask*255).astype(int)
        np_image = (np_image*255).astype(int)
        
        # If quantiles is enable then calculate it and use it instead based on the upper bound
        if self.use_quantile_bounds:
            _upper_bound = np.quantile(image_mask.ravel(), self.upper_bound)
            #_lower_bound = np.quantile(image_mask.ravel(), lower_bound)
            #print("Quantile bounds",upper_bound,lower_bound)
        else:
            _upper_bound = self.upper_bound
            #_lower_bound = lower_bound
        # Remove background from mask
        # Below 10 exists due to some issues where background is not zero
        image_mask[np_image == 0] = 0
        
        segmented_mask = segment_mask(np_image, image_mask, upper_bound=_upper_bound)
        
        return segmented_mask, image_mask,class_idx,_upper_bound
                
    def features(self, i, image_name, nifti_image, patient_class, observe_class):
        """
        Calculate score and return info of the segmented features
        Args:
            i: index
            image_name: Name of image
            nifti_image: image from nifti
            patient_class: Patient class
            observe_class: Which class to observe
            
        """
        #i, image_name, nifti_image, patient_class, observe_class = data
        
        segmented_mask, image_mask, class_idx,_upper_bound = self.extract(nifti_image, observe_class)
        
        #display(segmented_mask)
        #print(f"Image: {image_name}, Patient: {patient_class}, Observe: {observe_class}, Model predict: {class_idx}", end='\r')
        # TODO is it bbox_area or area
        features = measure.regionprops(segmented_mask, intensity_image=image_mask)
        new_features = {
            'bbox_area':[feature.bbox_area for feature in features], 
            'mean_intensity':[feature.mean_intensity for feature in features], 
            'bbox':bounding_boxes(features),
            'upper_bound':_upper_bound,
            'use_quantile_bounds':self.use_quantile_bounds,
        }
        new_features.update({

            'score':self.lambda1 * np.mean(new_features['mean_intensity'])/np.max(image_mask) - self.lambda2*((np.array(new_features['bbox_area'])+self.min_area)/image_mask.size)
        })
        
        #del class_scores, nifti_image, segmented_mask, image_mask, features
        return {'image':image_name, 'patient_class':patient_class, 'observe_class':observe_class, 'probability_class':class_idx, **new_features}