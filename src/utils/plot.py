import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from math import ceil
import pytorch_lightning as pl
import seaborn as sns

from src.utils import preprocess
from src.utils.cmap import parula_map

def intensity_distribution(image, title=""):
    """Plot the intensity distribution of an input image"""
    fig = plt.figure()
    ax = sns.histplot(image) #, image.max()
    # Ignore the first value as it is only zeros
    #_, counts = np.unique(image, return_counts=True)
    plt.xlim([0,image.max()])
    #plt.ylim([0,counts[1:].max()]) # Existed if we want to ignore zeros but caused a lot of trouble..
    plt.title(title)
    plt.xlabel("Intensity")
    plt.xlabel("Frequency")

def display_3D(im3d:np.ndarray, cmap:str="jet", step:int=2, plottype:str='imshow'):
    """Plot 3D image as 2D slices"""
    
    ncols = 9
    nrows = 1 if ceil(im3d.shape[0]/(ncols*step)) == 0 else ceil(im3d.shape[0]/(ncols*step))
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 1*nrows))

    vmin = im3d.min()
    vmax = im3d.max()
    
    flatten_axes = axes.flatten()
    for i, image in enumerate(im3d[::step]):
        if plottype == 'imshow':
            flatten_axes[i].imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        elif plottype == 'hist':
            flatten_axes[i].hist(image.ravel(), bins=image.max(), histtype='step', color='black');
            
        flatten_axes[i].set_xticks([])
        flatten_axes[i].set_yticks([])
        
    for ax in flatten_axes[len(im3d[::step]):]:
        ax.set_visible(False)
        
    return fig

def features_regions(bboxes:list, image_mask:np.ndarray,step=1, plot_title=""):
    """Plot the extracted features"""
    ncols = 9
    nrows = 1 if len(image_mask) == 0 else int(np.ceil(len(image_mask)/ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 1*nrows))

    vmin = image_mask.min()
    vmax = image_mask.max()
    fig.suptitle(plot_title)
    flatten_axis = axes.flatten()
    for ax, image in zip(flatten_axis, image_mask[::step]):
        ax.imshow(image, cmap='jet', vmin=vmin, vmax=vmax)

        ax.set_xticks([])
        ax.set_yticks([])

    # Add boundaries
    
    for x0,y0,x1,y1,z0,z1 in bboxes:
        for z in range(z0,z1):
            flatten_axis[z].add_patch(mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0,fill=False, edgecolor='red', linewidth=2))
    
    for ax in flatten_axis[len(image_mask[::step]):]:
        ax.set_visible(False)
        
    return fig

def center_distribution(bbox_coords):
    """Plot the distribution"""
    bbox_listed = list(zip(*bbox_coords))

    fig, axes = plt.subplots(1, 3, figsize=(10,5))
    fig.suptitle("Distribution of the center of each bounding box for x,y,z")
    for ax,cord in zip(axes.flatten(),combine_coordinates(np.array(bbox_listed))):
        sns.histplot(cord, ax=ax,bins=10)

def apply_image_bboxes(image,bbox_coords):
    """
    Takes coordinates of bounding boxes and plot them.
    
    Args:
        bbox_coords (list[int]): Expect shape of (x0,y0,x1,y1,z0,z1)
    """

    for feature in features:
        x0,y0,x1,y1,z0,z1 = bounding_boxes(feature)
        for z in range(z0,z1):
            flatten_axis[z].add_patch(mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0,fill=False, edgecolor='red', linewidth=2))
            
    return image

def ROC(roc_classes, prefix='', fig=None):
    # Returns (auc, fpr, tpr), roc_fig
    fig = plt.figure(figsize = (10,7))
    lw = 2
    colors = np.array(['aqua', 'darkorange', 'cornflowerblue'])
    fpr, tpr, threshold = roc_classes
    
    metric_list = np.zeros(3)
    for i in range(len(roc_classes)):
        auc = preprocess.tensor2numpy(pl.metrics.functional.auc(fpr[i],tpr[i]))
        
        _fpr = preprocess.tensor2numpy(fpr[i])
        _tpr = preprocess.tensor2numpy(tpr[i])
        
        metric_list[0]+=auc
        metric_list[1]+=_fpr.mean()
        metric_list[2]+=_tpr.mean()
        
        plt.plot(_fpr, _tpr, color=colors[i], lw=lw,
                 label='ROC curve of class {0} (area={1:0.2f} tpr={2:0.2f} fpr={3:0.2f})'
                 ''.format(i,auc, _tpr.mean(), 1-_fpr.mean())) #
    
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    
    metric_list = metric_list/3

    return metric_list, fig

def imshow(image, cmap=parula_map, figsize=(8,4),colormap=False,colormap_shrink=1, disable_axis=True):
    fig = plt.figure(figsize=figsize)
    im = plt.imshow(image, cmap=cmap)
    
    if colormap:
        #plt.subplots_adjust(wspace=0.01, hspace=0)
        cbar = fig.colorbar(im,shrink=colormap_shrink,pad=0.01)
    if disable_axis:
        plt.axis('off')

    plt.show()

    