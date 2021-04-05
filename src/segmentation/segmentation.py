from skimage import exposure, filters, measure
from scipy import ndimage
import matplotlib.patches as mpatches
import numpy as np
import matplotlib.pyplot as plt
import torch
#from src.segmentation.roi import RoIAlign
from src.segmentation.roi_align.roi_align import RoIAlign
from src.utils.preprocess import tensor2numpy

from skimage import exposure, io, util
from skimage import data, img_as_float
from skimage.morphology import disk
from skimage.filters import rank

def intensity_distribution(image, title=""):
    """Plot the intensity distribution of an input image"""
    b, bins, patches = plt.hist(image, 255)
    # Ignore the first value as it is only zeros
    _, counts = np.unique(image, return_counts=True)
    plt.xlim([0,255])
    plt.ylim([0,counts[1:].max()])
    plt.title(title)
    plt.show()
    

def mask_mean_filter(mask:np.ndarray) -> np.ndarray:
    """Apply a mean filter of size (7,7,7) on the 3D image"""
    mean_mask = ndimage.median_filter(mask, size=(7,7,7))
    
    return mean_mask

def mask_logarithmic_scale(mask:np.ndarray) -> np.ndarray:
    """Convert mask to logarithmic scale"""
    logarithmic_corrected = exposure.adjust_log(mask, 1)
    
    return logarithmic_corrected

def mask_clipping(mask:np.ndarray) -> np.ndarray:
    """Remove intensity outside of procentual boundaries"""
    vmin, vmax = np.percentile(mask, q=(20, 99.5))

    clipped_data = exposure.rescale_intensity(
        mask,
        in_range=(vmin, vmax),
        out_range=np.float32
    )
    return clipped_data

def display(im3d:np.ndarray, cmap:str="jet", step:int=2, plottype:str='imshow'):
    """Plot 3D image as 2D slices"""
    
    ncols = 9
    nrows = 1 if im3d.shape[0]//(ncols*step) == 0 else im3d.shape[0]//(ncols*step)
    
    _, axes = plt.subplots(nrows=nrows, ncols=9, figsize=(10, 1*nrows))

    vmin = im3d.min()
    vmax = im3d.max()
    
    for ax, image in zip(axes.flatten(), im3d[::step]):
        if plottype == 'imshow':
            ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        elif plottype == 'hist':
            ax.hist(image.ravel(), bins=255, histtype='step', color='black');
            
        ax.set_xticks([])
        ax.set_yticks([])
        
def remove_known_background_from_mask(image:np.ndarray, image_mask:np.ndarray) -> np.ndarray:
    """We know that background cant be useful activations. Remove them"""
    im = image_mask.copy() 
    im[image == 0] = 0
    return im

def check_join_segmentations(mask_one:np.ndarray, mask_two:np.ndarray) -> np.ndarray:
    """Observe the intersection of two masks"""
    from skimage.segmentation import join_segmentations
    return join_segmentations(mask_one, mask_two)

def segment_mask(image_mask:np.ndarray) -> np.ndarray:
    """Simple segmentation of the mask input
    Note: This one might need some scientific rework since it is not very good
    """
    n=32
    im = filters.gaussian(image_mask, sigma=1/(4.*n), mode='nearest')
    mask = im > im.max()*0.6
    label_im, nb_labels = ndimage.label(mask)
    return label_im

def extract_features(segmentation:np.ndarray, image_mask:np.ndarray):
    """Create objects of each slice where the selected regions are extract
    
    Args:
        segmentation (np.ndarray): A segmented image where the boundaries of each class is clearly defined
        image_mask (np.ndarray): An image mask where the intensities is observed.
        
    Return:
        output: List[RegionProperties]
        """
    #features = []
    #for seg, mask in zip(segmentation, image_mask):
    return measure.regionprops(segmentation, intensity_image=image_mask)  # only one object
        #features.append(props)
    #return features

def bounding_boxes(features):
    """Expects features from 2D images
    
    Args:
        features (List[RegionProperties])
    Return:
        output List[K, 6] with the box coordinates in (y0, x0, y1, x1, z0, z1) and k is batch size.
    """
    boxes = []
    if isinstance(features, list):
    # Features must exist!
        for feature in features:

            # For each bounding box:
            # Wild guess that z0 and z1 is correct. x0 and x1 positions cant be depth

            y0, x0, z0, y1, x1,z1 = feature.bbox
            boxes.append((y0, x0, y1, x1,z0,z1))
    else:
        #y0, x0, z0, y1, x1,z1 = features.bbox
        z0, y0, x0, z1, y1, x1 = features.bbox
        print(features)
        return (y0, x0, y1, x1,z0,z1)
    return boxes

def plot_features_regions(features:list, image_mask:np.ndarray,step=1):
    """Plot the extracted features"""
    ncols = 9
    nrows = 1 if len(image_mask) == 0 else int(np.ceil(len(image_mask)/ncols))
    _, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 1*nrows))

    vmin = image_mask.min()
    vmax = image_mask.max()

    for ax, image in zip(axes.flatten(), image_mask[::step]):
        ax.imshow(image, cmap='jet', vmin=vmin, vmax=vmax)

        ax.set_xticks([])
        ax.set_yticks([])
        
    # Add boundaries
    flatten_axis = axes.flatten()
    for feature in features:
        y0, x0, y1, x1,z0, z1 = bounding_boxes(feature)
        rect = mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0,fill=False, edgecolor='red', linewidth=2)
        print("asdasd",z0,z1, feature.bbox)
        for z in range(z0,z1):
            print(z)
            #flatten_axis[z].add_patch(rect)
            flatten_axis[z].patches.extend([rect])
        print(feature.centroid)
        #Note check that these coordinates are correct!
        flatten_axis[int(feature.centroid[0])].plot(*feature.centroid[::-1], marker='x', color='r')

def roi_align(image, features):
    """ Create aligned image rois for the neural network
    Arg:
        * Image of shape Tuple[D,H,W]
        * List of features of shape List[Tuple[int,int,int,int,int]] 
    """
    image_tensor = torch.from_numpy(image).unsqueeze(1).float()
    boxes = torch.Tensor(bounding_boxes(features))
    
    roialign = RoIAlign((40,40,10),spatial_scale=1.0,sampling_ratio=-1).cuda()
    image_rois = roialign.forward(image_tensor.cuda(),boxes.cuda())
    display(tensor2numpy(image_rois).squeeze(1),step=1)
    
    return image_rois

def sequential_processing(image:np.ndarray, image_mask:np.ndarray) -> None:
    """Run each processing of mask to segmentation"""
    
    # Remove mask that we know are a background (not a part of the brain scan)
    mask_no_background = remove_known_background_from_mask(image, image_mask)
    
    # Rescale mask and segment it
    mask_mean = mask_logarithmic_scale(mask_no_background)
    segmented_mask = segment_mask(mask_mean)
    
    # Extract features with the intensities of the mask with removed background and plot it
    features = extract_features(segmented_mask, mask_no_background)
    plot_features_regions(features, mask_no_background)
    
    return features