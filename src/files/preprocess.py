"""
Preprocessing functions for images, labels
"""



import numpy as np
import torch
from torch import Tensor, from_numpy
#from torch.nn.functional import pad
from typing import Union, Tuple
from skimage.transform import resize
import torchvision


def normalize(x:Union[Tensor,np.ndarray]) -> Union[Tensor,np.ndarray]:
    """Min-max normalization (0-1):

    Parameters
    ----------
    x : Union[Tensor,np.ndarray]
        Can be tensor or ndarray to be converted

    Returns
    -------
    Union[Tensor,np.ndarray]
        * Return same type as input but scaled between 0 - 1

    
    """
    return (x - x.min())/(x.max()-x.min())

def batchisize_to_5D(x:Union[Tensor,np.ndarray]) -> Union[Tensor,np.ndarray]:
    """Takes an input with lower dimensions than 5 and pad the dimensions to 5. len(x.shape) < 5 -> len(x.shape) == 5

    Parameters
    ----------
    x : Union[Tensor,np.ndarray]
        Input vector

    Returns
    -------
    Union[Tensor,np.ndarray]
        x with len(x.shape) == 5

    
    """
    if isinstance(type(x), Tensor):
        return x.expand((*[1]*(5-len(x.shape)),*[-1]*len(x.shape)))
    else:
        return np.expand_dims(x,[i for i,_ in enumerate(range(5-len(x.shape)))])
    
def folder2labels(images:list, classes:dict) -> np.ndarray: #, delimiter:str
    """Extract labels from filename based on delimiter.
    Expects a label before filename
    
    Example: /AD/image_file.png

    Parameters
    ----------
    images : list
        list containing all filenames
    classes : dict
        The allowed classes that can be used for labels

    Returns
    -------
    np.ndarray
        A list of labels based on the given classes from the image filenames.

    
    """
    #if delimiter not in images[0]: raise ValueError(f"The defined delimiter could not be found in image input! Image input: {images[0]}")
    return np.array([classes[img_path.rsplit("/",2)[1]] for img_path in images])#np.array([classes[img_path.rsplit("/",1)[1].split(delimiter,1)[0]] for img_path in images])

def mask_threshold(image:np.ndarray, threshold:float) -> None:
    """Place a threshold on image and remove all values that are below the threshold.
    All values under the threshold is set to 0

    Parameters
    ----------
    image : np.ndarray
        Input image that threshold should be applied to.
    threshold : float
        Value between 0-1 at which procent of the max value that should be taken as a mask

    Returns
    -------

    
    """
    img_threshold = image.max()*threshold
    image[image < img_threshold] = 0
    return image #np.ma.masked_where(image <= img_threshold,image)

def set_outside_of_scan_pixels_2_zero(image:np.ndarray, mask:np.ndarray) -> np.ndarray:
    """Everything that is zero outside of image is set to zero on the mask. Returns the modified mask

    Parameters
    ----------
    image : np.ndarray
        
    mask : np.ndarray
        

    Returns
    -------

    
    """
    return np.ma.masked_where(image == 0,mask)

def image2axial(image:np.ndarray) -> np.ndarray:
    """Modify image to display axial view. Expects an image of (B,H,W) or (B,C,H,W)

    Parameters
    ----------
    image : np.ndarray
        

    Returns
    -------

    
    """
    if not isinstance(image, np.ndarray): raise TypeError(f"Expected np.ndarray. Got: {type(image)}")
    if len(image.shape) == 3:
        return np.flip(image.transpose(2,1,0),axis=1).copy()
    #elif len(image.shape) == 4:
    #    return np.flip(image.transpose(3,1,2,0))
    else:
        raise ValueError(f"Expected length of 3 or 4. Got: {len(image.shape)}")

def tensor2numpy(data:torch.tensor):
    """Send to CPU. If computational graph is connected then detach it as well.

    Parameters
    ----------
    data : torch.tensor
        

    Returns
    -------

    
    """
    if data.requires_grad:
        return data.detach().cpu().numpy()
    else:
        return data.cpu().numpy()

def uint8(image:np.ndarray):
    """

    Parameters
    ----------
    image : np.ndarray
        

    Returns
    -------

    
    """
    # Change range 0-1 to 0-255 and change type to uint8
    return (image*255).astype(np.uint8) 
                   
def preprocess_image(image:np.ndarray, input_shape:Tuple=(79,95,79), normalized=True) -> Tensor:
    """Resize, normalize between 0-1 and convert to uint8

    Parameters
    ----------
    image : np.ndarray
        
    input_shape : Tuple
        (Default value = (79,95,79))
    normalized :
        (Default value = True)

    Returns
    -------

    
    """
    
    if input_shape != image.shape:
        image = resize(image,input_shape)
    
    if normalized: 
        image = normalize(image)

    return image

def to_grid(image:np.ndarray, max_num_slices:int=None,pad_value:float=0.5, nrow:int=10, padding:int=2) -> Tensor:
    """Create grid from image based on maximum number of slices.

    Parameters
    ----------
    image : np.ndarray
        Image with multiple slices of shape Tuple[D,H,W]
    max_num_slices : int
        The number of slices the image should be reduced. (Default value = None)
    pad_value : float
        (Default value = 0.5)
    nrow : int
        (Default value = 10)
    padding : int
        (Default value = 2)

    Returns
    -------
    Tensor
        A grid image with reduced number of slices.

    
    """
    assert len(image.shape) == 3
    
   
    image=normalize(image)
    
    if max_num_slices != None:
        image = np.stack([np.mean(x,axis=0) for x in np.array_split(image, max_num_slices)]) #greedy_split(image,max_num_slices)
    
    plt_image = from_numpy(image).float().unsqueeze(1)

    # Convert to grid 
    grid_image = torchvision.utils.make_grid(plt_image, nrow=nrow,pad_value=pad_value,padding=2)[0]
    
    return grid_image