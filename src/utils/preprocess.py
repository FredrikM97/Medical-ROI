import numpy as np
import torch
from torch import Tensor, from_numpy
#from torch.nn.functional import pad
from typing import Union, Tuple
from skimage.transform import resize
import torchvision


def normalize(x:Union[Tensor,np.ndarray]) -> Union[Tensor,np.ndarray]:
    """ Min-max normalization (0-1):
    
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
    if delimiter not in images[0]: raise ValueError(f"The defined delimiter could not be found in image input! Image input: {images[0]}")
    return np.array([classes[img_path.rsplit("/",1)[1].split(delimiter,1)[0]] for img_path in images])

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
        return np.flip(image.transpose(2,1,0),axis=1).copy()
    #elif len(image.shape) == 4:
    #    return np.flip(image.transpose(3,1,2,0))
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


def tensor2numpy(data):
    """Send to CPU. If computational graph is connected then detach it as well."""
    if data.requires_grad:
        return data.detach().cpu().numpy()
    else:
        return data.cpu().numpy()

def uint8(image:np.ndarray):
    # Change range 0-1 to 0-255 and change type to uint8
    return (image*255).astype(np.uint8) 
                   
def preprocess_image(image:np.ndarray, input_shape:Tuple=(79,95,79), normalized=True) -> Tensor:
    """Resize, normalize between 0-1 and convert to uint8"""

    #if not isinstance(image, np.ndarray): raise ValueError(f"Expected image to be ndarray. Got: {type(image)}")
    #print("Unioquew", np.unique(image))
    
    # Testing to zero-pad instead of resize
    #print(input_shape, image.shape, input_shape - list(image.shape))
    ##print([x-y for x,y in zip(input_shape, image.shape)]) #[x-y for x,y in zip(input_shape, image.shape)]
    #print("Asdasd", input_shape, "Image shape", image.shape)
    
    
    #pad_shape = [(x-y)*2 for x,y in zip(input_shape, image.shape)]
    #print(pad_shape, input_shape)
    
    #print("asdasdasd",image.dtype)
    #if padded:      
    #    new_image = pad(image, input_shape)
    #if resized:
    #print("image type",type(image))
    
    image = resize(image,input_shape)
    
    
    #print(image.max(),image.min())
    #print(image.shape)
    if normalized: 
        image = normalize(image)
    #image = uint8(image)
    #print("image type2",type(image))
    return image

def to_grid(image:np.ndarray, max_num_slices=None,pad_value=0.5, nrow=10) -> Tensor:
    """ Create grid from image based on maximum number of slices.

    Args:
        * image: Image with multiple slices of shape Tuple[D,H,W]
        * max_num_slices: The number of slices the image should be reduced.

    Return:
        * Return a grid image with reduced number of slices.

    """
    #            plot.display_3D(image_process(self.activation_map(class_idx, class_scores)))
    assert len(image.shape) == 3
    
   
    image=normalize(image)
    
    if max_num_slices != None:
        image = np.stack([np.mean(x,axis=0) for x in greedy_split(image,max_num_slices)])
    
    plt_image = from_numpy(image).float().unsqueeze(1)

    # Convert to grid 
    grid_image = torchvision.utils.make_grid(plt_image, nrow=nrow,pad_value=pad_value)[0]
    return grid_image