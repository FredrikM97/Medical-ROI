import numpy as np
from torch import Tensor
from typing import Union

def normalize(x:Union[Tensor,np.ndarray]) -> Union[Tensor,np.ndarray]:
    """ Min-max normalization (0-1)
    
    Args:
        * x - Can be tensor or ndarray to be converted 
    
    Return:
        * Return same type as input but scaled between 0 - 1
    """
    return (x - x.min())/(x.max()-x.min())

def batchisize_to_5D(x:Union[Tensor,np.ndarray]) -> Union[Tensor,np.ndarray]:
    """ Takes an input with lower dimensions than 5 and pad the dimensions to 5. len(x.shape) < 5 -> len(x.shape) == 5
    
    Args:
        * x - Input vector
        
    Return:
        * x with len(x.shape) == 5
    """
    if isinstance(type(x), Tensor):
        return x.expand((*[1]*(5-len(x.shape)),*[-1]*len(x.shape)))
    else:
        return np.expand_dims(x,[i for i,_ in enumerate(range(5-len(x.shape)))])
    
def filename2labels(images:list, classes:dict, delimiter:str) -> np.ndarray:
    """ Extract labels from filename based on delimiter. 
    
    Args:
        * images: list containing all filenames
        * classes: The allowed classes that can be used for labels
        * delimiter: The suggested delimiter that should seperate items
        
    Return:
        * Return a list of labels based on the given classes from the image filenames.
    
    """
    assert delimiter in images[0], "The defined delimiter could not be found in image input!"
    return np.array([classes[img_path.rsplit("/",1)[1].split(delimiter,1)[0]] for img_path in images])
<<<<<<< HEAD
=======

def mask_threshold(image:np.ndarray, threshold:float) -> None:
    """Place a threshold on image and remove all values that are below the threshold.
    All values under the threshold is set to 0
 
    
    Args:
        * image: Input image that threshold should be applied to.
        * treshold: Value between 0-1 at which procent of the max value that should be taken as a mask
    """
    img_threshold = image.max()*threshold
    image[image < img_threshold] = 0
    return image #np.ma.masked_where(image <= img_threshold,image)

def set_outside_of_scan_pixels_2_zero(image:np.ndarray, mask:np.ndarray) -> np.ndarray:
    """Everything that is zero outside of image is set to zero on the mask. Returns the modified mask"""
    return np.ma.masked_where(image == 0,mask)

def image2axial(image:np.ndarray) -> np.ndarray:
    """Modify image to display axial view. Expects an image of (B,H,W) or (B,C,H,W) """
    if not isinstance(image, np.ndarray): raise TypeError(f"Expected np.ndarray. Got: {type(image)}")
    if len(image.shape) == 3:
        return image.transpose(2,1,0)
    elif len(image.shape) == 4:
        return image.transpose(3,1,2,0)
    else:
        raise ValueError(f"Expected length of 3 or 4. Got: {len(image.shape)}")

def greedy_split(arr:np.ndarray, n:int, axis=0) -> list:
    """Split an image of slices into smaller slices. Example: 1x79 slices with an n of 16 -> 5x16 slices where 1 slice contains 5 of the 79 slices."""
    assert isinstance(arr, np.ndarray), f"Expected np.ndarray, got: {type(arr)}"
    length = arr.shape[axis]

    # compute the size of each of the first n-1 blocks
    block_size = np.ceil(length / float(n))

    # the indices at which the splits will occur
    ix = np.arange(block_size, length, block_size).astype(np.uint8)
    return np.split(arr, ix, axis)

<<<<<<< HEAD
>>>>>>> Bug fixes and cleanup
=======
def tensor2numpy(data):
    """Send to CPU. If computational graph is connected then detach it as well."""
    if data.requires_grad:
        return data.detach().cpu().numpy()
    else:
        return data.cpu().numpy()
>>>>>>> Bug fixes for modules
