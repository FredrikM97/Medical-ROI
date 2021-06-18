"""
Plot functions for images
"""




import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from math import ceil
import pytorch_lightning as pl
import seaborn as sns
import cv2
from matplotlib import colors
from math import ceil, floor

from src.files import preprocess
from src.display.cmap import parula_map

#sns.set(font_scale=2.5)

def intensity_distribution(image:np.ndarray, title:str=""):
    """Plot the intensity distribution of an input image

    Parameters
    ----------
    image : np.ndarray
        
    title : str
        (Default value = "")

    Returns
    -------

    
    """
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
    """Plot 3D image as 2D slices

    Parameters
    ----------
    im3d : np.ndarray
        
    cmap : str
        (Default value = "jet")
    step : int
        (Default value = 2)
    plottype : str
        (Default value = 'imshow')

    Returns
    -------

    
    """
    
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

def features_regions(bboxes:list, image_mask:np.ndarray,grid_kwgs:dict={}):
    """Plot the extracted features

    Parameters
    ----------
    bboxes : list
        
    image_mask : np.ndarray
        
    grid_kwgs : dict
        (Default value = {})

    Returns
    -------

    
    """

    vmin = image_mask.min()
    vmax = image_mask.max()
    
    empty_array = np.zeros(image_mask.shape)
    
    for tmp_bbox in bboxes:
        start = (tmp_bbox[0],tmp_bbox[1])
        end = (tmp_bbox[2],tmp_bbox[3])
        for slice_idx,x in enumerate(empty_array[tmp_bbox[4]:tmp_bbox[5]], start=tmp_bbox[4]):
            x = cv2.rectangle(x, start, end,(255,0,0),2) 
    tmp_gridded =preprocess.to_grid(empty_array,pad_value=0, **grid_kwgs.copy()) 

    masked = np.ma.masked_where(tmp_gridded == 0, tmp_gridded)
    
    #fig = plt.figure(figsize=(8,8))
    plt.imshow(preprocess.to_grid(image_mask, **grid_kwgs.copy()), 'gray', interpolation='none', vmin=vmin, vmax=vmax)
    plt.imshow(masked,cmap=colors.ListedColormap(['red']), interpolation='none', alpha=1, vmin=vmin, vmax=vmax)
    #plt.tight_layout()
    
    plt.axis('off')

def center_distribution(bbox_coords:list) -> None:
    """Plot the distribution

    Parameters
    ----------
    bbox_coords : list
        

    Returns
    -------

    
    """
    bbox_listed = list(zip(*bbox_coords))

    fig, axes = plt.subplots(1, 3, figsize=(10,5))
    fig.suptitle("Distribution of the center of each bounding box for x,y,z")
    for ax,cord in zip(axes.flatten(),combine_coordinates(np.array(bbox_listed))):
        sns.histplot(cord, ax=ax,bins=10)

def apply_image_bboxes(image,bbox_coords):
    """Takes coordinates of bounding boxes and plot them.

    Parameters
    ----------
    image :
        
    bbox_coords : list[int]
        Expect shape of (x0,y0,x1,y1,z0,z1)

    Returns
    -------

    
    """

    for feature in features:
        x0,y0,x1,y1,z0,z1 = bounding_boxes(feature)
        for z in range(z0,z1):
            flatten_axis[z].add_patch(mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0,fill=False, edgecolor='red', linewidth=2))
            
    return image

def roc(roc_classes, prefix:str='', fig=None):
    """

    Parameters
    ----------
    roc_classes :
        
    prefix : str
        (Default value = '')
    fig :
        (Default value = None)

    Returns
    -------

    
    """
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

def precision_recall_curve(precision, recall):
    """

    Parameters
    ----------
    precision :
        
    recall :
        

    Returns
    -------

    
    """
    fig = plt.figure(figsize = (10,7))
    lw=2
    colors = np.array(['aqua', 'darkorange', 'cornflowerblue'])
    #for i in range(len(roc_classes)):
    metric_list = np.zeros(2)
    for i in range(len(precision)):
        _pr = preprocess.tensor2numpy(precision[i])
        _re = preprocess.tensor2numpy(recall[i])
        
        metric_list[0]+=_pr.mean()
        metric_list[1]+=_re.mean()
        
        plt.plot(_re,_pr, color=colors[i],lw=lw,
                 label='Precision recall curve of class {0} (precision={1:0.2f} recall={2:0.2f})'
                 ''.format(i, _pr.mean(), _re.mean())) #
    
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower right")
    
    metric_list = metric_list/2

    return metric_list, fig

def confusion_matrix(cm):
    """

    Parameters
    ----------
    cm :
        

    Returns
    -------

    
    """
    with sns.plotting_context("talk", font_scale=1.5):
        #fig = plt.figure(figsize=(20,20))
        f, axs = plt.subplots(1, 1, figsize=(20,20))
        sns.heatmap(preprocess.tensor2numpy(cm), annot=True, vmin=0, vmax=1, xticklabels=['CN','MCI','AD'], yticklabels=['CN','MCI','AD'], ax=axs) #, annot_kws={"size": 20}
        axs.set_xlabel("Predicted label")
        axs.set_ylabel("True label")
    
    return f

def imshow(image, cmap=parula_map, figsize=(8,4),colormap:bool=False,colormap_shrink:float=1, disable_axis:bool=True):
    """

    Parameters
    ----------
    image :
        
    cmap :
        (Default value = parula_map)
    figsize :
        (Default value = (8,4))
    colormap : bool
        (Default value = False)
    colormap_shrink : float
        (Default value = 1)
    disable_axis : bool
        (Default value = True)

    Returns
    -------

    
    """
    fig = plt.figure(figsize=figsize)
    im = plt.imshow(image, cmap=cmap)
    
    if colormap:
        #plt.subplots_adjust(wspace=0.01, hspace=0)
        cbar = fig.colorbar(im,shrink=colormap_shrink,pad=0.01)
    if disable_axis:
        plt.axis('off')

    plt.show()

  
def advanced_plot(slices):
    """

    Parameters
    ----------
    slices :
        

    Returns
    -------

    
    """
    "Advanced plotting function"
    import matplotlib.gridspec as gridspec
    
    w = 20
    h = 10
    columns = 4 if len(slices) >= 4 else len(slices)
    rows = floor(len(slices)/columns)
    
    fig = plt.figure(figsize=(w, h))

    
    gs1 = gridspec.GridSpec(rows,columns)
    gs1.update(wspace=0.025, hspace=0.05) 

    ax = []
    
    for i in range(columns*rows):
        #fig.set_size_inches(w, h)
        # create subplot and append to ax
        ax1 = fig.add_subplot(gs1[i])
        #ax1.set_aspect('equal')

        ax.append(ax1)
        #ax[-1].set_title("ax:"+str(i))  # set title
        plt.imshow(slices[i], cmap="gray", origin="lower")
    
    for ax in fig.get_axes():
        ax.label_outer()

    plt.show()  # finally, render the plot

def meta_settings(rows:int=1, cols:int=2, figsize=(16,16)):
    """

    Parameters
    ----------
    rows : int
        (Default value = 1)
    cols : int
        (Default value = 2)
    figsize :
        (Default value = (16,16))

    Returns
    -------

    
    """
    "Plot settings of meta data"
    plt.rcParams.update({'font.size': 15})
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=figsize)
    fig.set_tight_layout(True)
    return fig, axes

def move_legend(ax, new_loc:str="upper left",title:str=None,**kws):
    """If the legend disapear this function should be used.
    Recommended from https://github.com/mwaskom/seaborn/issues/2280 where matplotlib have some weird implementation which makes the legend disappear..

    Parameters
    ----------
    ax :
        
    new_loc : str
        (Default value = "upper left")
    title : str
        (Default value = None)
    **kws :
        

    Returns
    -------

    
    """
    if (old_legend := ax.legend_) is not None:
        handles = old_legend.legendHandles
        labels = [t.get_text() for t in old_legend.get_texts()]
        if not title: title = old_legend.get_title().get_text() #, loc=new_loc
            
        ax.legend(handles, labels, title=title, borderaxespad=0., bbox_to_anchor=(1.02, 1), loc=new_loc,**kws)

def set_plot_settings(ax,rotation:int=0, title:str=None, xlabel:str=None, ylabel:str=None, **kws):
    """

    Parameters
    ----------
    ax :
        
    rotation : int
        (Default value = 0)
    title : str
        (Default value = None)
    xlabel : str
        (Default value = None)
    ylabel : str
        (Default value = None)
    **kws :
        

    Returns
    -------

    
    """
    sns.set_theme(style="ticks", color_codes=True)
    if title: ax.set_title(title)
    if xlabel: ax.set(xlabel=xlabel)
    if ylabel: ax.set(ylabel=ylabel)
    
    ax.tick_params(axis='x',which='major', rotation=rotation)
    
    
def countplot(
                df, 
                x:str=None, 
                y:str=None, 
                hue:str=None, 
                order:list=None, 
                ax=None,
                plot_kws:dict={},
                setting_kws:dict={},
                legend_kws:dict={},
                **kws
            ):
    """Example:
    dplay.do_countplot(
        df,
        x="subject.visit.assessment.component.assessmentScore_FAQTOTAL",
        hue="subject.researchGroup",
        order=np.arange(1,31),
        rotation=90,
        xlabel="FAQTOTAL",
        ylabel="Frequency",
        title="Functional Activities Questionnaires (FAQTOTAL)",
        ax=axes[0,0]
    )

    Parameters
    ----------
    df :
        
    x : str
        (Default value = None)
    y : str
        (Default value = None)
    hue : str
        (Default value = None)
    order : list
        (Default value = None)
    ax :
        (Default value = None)
    plot_kws : dict
        (Default value = {})
    setting_kws : dict
        (Default value = {})
    legend_kws : dict
        (Default value = {})
    **kws :
        

    Returns
    -------

    
    """
    sns.set(font_scale=1.1)

    ax1 = sns.countplot(
                x=x,
                y=y,
                hue = hue, 
                data = df,
                order=order,
                ax=ax,
                **plot_kws
            )

    set_plot_settings(ax1,**setting_kws)
    move_legend(ax1,**legend_kws)
    plt.tight_layout()
    return ax1

def histplot(
            df, 
            x=None,
            y=None,
            hue:str=None, 
            multiple='stack',
            bins:int='auto',
            discrete=True,
            ax=None,
            plot_kws:dict={},
            setting_kws:dict={},
            legend_kws:dict={},
            **kws
        ):
    """

    Parameters
    ----------
    df :
        
    x :
        (Default value = None)
    y :
        (Default value = None)
    hue : str
        (Default value = None)
    multiple :
        (Default value = 'stack')
    bins : int
        (Default value = 'auto')
    discrete :
        (Default value = True)
    ax :
        (Default value = None)
    plot_kws : dict
        (Default value = {})
    setting_kws : dict
        (Default value = {})
    legend_kws : dict
        (Default value = {})
    **kws :
        

    Returns
    -------

    
    """
    sns.set(font_scale=1.1)
    ax1 = sns.histplot(
        df,
        x=x,
        y=y, 
        hue=hue,
        bins=bins,
        discrete=discrete,
        multiple=multiple,
        ax=ax,
        **plot_kws,
        
    )
    set_plot_settings(ax1,**setting_kws)
    move_legend(ax1,**legend_kws)
    plt.tight_layout()
    return ax1

def catplot(
            df, 
            x=None,
            y=None,
            hue:str=None, 
            kind='bar',
            multiple='stack',
            bins:int='auto',
            discrete=True,
            plot_kws:dict={},
            setting_kws:dict={},
            legend_kws:dict={},
            **kws
        ):
    """

    Parameters
    ----------
    df :
        
    x :
        (Default value = None)
    y :
        (Default value = None)
    hue : str
        (Default value = None)
    kind :
        (Default value = 'bar')
    multiple :
        (Default value = 'stack')
    bins : int
        (Default value = 'auto')
    discrete :
        (Default value = True)
    plot_kws : dict
        (Default value = {})
    setting_kws : dict
        (Default value = {})
    legend_kws : dict
        (Default value = {})
    **kws :
        

    Returns
    -------

    
    """
    sns.set(font_scale=1.1)
    ax1 = sns.catplot(
        df,
        x=x,
        y=y, 
        kind=kind,
        hue=hue,
        bins=bins,
        discrete=discrete,
        multiple=multiple,
        **plot_kws,
        
    )
    set_plot_settings(ax1,**setting_kws)
    move_legend(ax1,**legend_kws)
    plt.tight_layout()
    return ax1

    
def boxplot(
                df, 
                x:str=None, 
                y:str=None, 
                hue:str=None, 
                order:list=None, 
                ax=None,
                plot_kws:dict={},
                setting_kws:dict={},
                legend_kws:dict={},
                **kws
            ):
    """

    Parameters
    ----------
    df :
        
    x : str
        (Default value = None)
    y : str
        (Default value = None)
    hue : str
        (Default value = None)
    order : list
        (Default value = None)
    ax :
        (Default value = None)
    plot_kws : dict
        (Default value = {})
    setting_kws : dict
        (Default value = {})
    legend_kws : dict
        (Default value = {})
    **kws :
        

    Returns
    -------

    
    """
 
    sns.set(font_scale=1.1)

    ax1 = sns.boxplot(
                x=x,
                y=y,
                hue = hue, 
                data = df,
                order=order,
                ax=ax,
                **plot_kws
            )

    set_plot_settings(ax1,**setting_kws)
    move_legend(ax1,**legend_kws)
    plt.tight_layout()
    return ax1

def lineplot(
            df, 
            x=None,
            y=None,
            hue:str=None, 
            title:str=None,
            ax=None,
            plot_kws:dict={},
            setting_kws:dict={},
            legend_kws:dict={},
            **kws
        ):
    """

    Parameters
    ----------
    df :
        
    x :
        (Default value = None)
    y :
        (Default value = None)
    hue : str
        (Default value = None)
    title : str
        (Default value = None)
    ax :
        (Default value = None)
    plot_kws : dict
        (Default value = {})
    setting_kws : dict
        (Default value = {})
    legend_kws : dict
        (Default value = {})
    **kws :
        

    Returns
    -------

    
    """
    "Evaluate if this works. Not tested"
    sns.set(font_scale=1.1)
    ax1 = sns.lineplot(
        data=df,
        x=x,
        y=y, 
        hue=hue,
        ax=ax,
        **plot_kws,
        
    )
    set_plot_settings(ax1,**setting_kws)
    move_legend(ax1,**legend_kws)
    plt.tight_layout()
    return ax1