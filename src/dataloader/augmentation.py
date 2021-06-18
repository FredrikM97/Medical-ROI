"""
Agmentation for the dataloader
"""

from skimage.transform import warp, AffineTransform, ProjectiveTransform#, rotate
from skimage.exposure import equalize_adapthist, equalize_hist, rescale_intensity, adjust_gamma, adjust_log, adjust_sigmoid
from skimage.filters import gaussian
from skimage.util import random_noise
import random

from scipy.ndimage import rotate

from torchvision import transforms
import numpy as np
def randRange(a:float, b:float):
    """Generate random float values in desired range

    Parameters
    ----------
    a :
        
    b :
        

    Returns
    -------

    
    """
    return np.random.rand() * (b - a) + a


def randomAffine(im:"ndarray") -> "ndarray":
    """Affine transformation with random scale, rotation, shear and translation parameters

    Parameters
    ----------
    im :
        

    Returns
    -------

    
    """
    tform = AffineTransform(scale=(randRange(0.75, 1.3), randRange(0.75, 1.3)),
                            rotation=randRange(-0.25, 0.25),
                            shear=randRange(-0.2, 0.2),
                            translation=(randRange(-im.shape[1]//10, im.shape[1]//10), 
                                         randRange(-im.shape[2]//10, im.shape[2]//10)))
    for i in range(im.shape[0]):
        im[i] = warp(im[i], tform.inverse, mode='reflect')
        
    return im

def randomCrop(im:"ndarray") -> "ndarray":
    """croping the image in the center from a random margin from the borders

    Parameters
    ----------
    im :
        

    Returns
    -------

    
    """
    margin = 1/10
    start = [int(randRange(0, im.shape[1] * margin)),
             int(randRange(0, im.shape[2] * margin))]
    end = [int(randRange(im.shape[1] * (1-margin), im.shape[1])), 
           int(randRange(im.shape[2] * (1-margin), im.shape[2]))]
    return im[start[0]:end[0], start[1]:end[1]]


def randomIntensity(im:"ndarray") -> "ndarray":
    """Rescales the intensity of the image to random interval of image intensity distribution

    Parameters
    ----------
    im :
        

    Returns
    -------

    
    """
    return rescale_intensity(im,
                             in_range=tuple(np.percentile(im, (randRange(0,10), randRange(90,100)))),
                             out_range=tuple(np.percentile(im, (randRange(0,10), randRange(90,100)))))

def randomGamma(im:"ndarray") -> "ndarray":
    """Gamma filter for contrast adjustment with random gamma value.

    Parameters
    ----------
    im :
        

    Returns
    -------

    
    """
    return adjust_gamma(im, gamma=randRange(0.95, 1.05))

def randomGaussian(im:"ndarray") -> "ndarray":
    """Gaussian filter for bluring the image with random variance.

    Parameters
    ----------
    im :
        

    Returns
    -------

    
    """
    return gaussian(im, sigma=randRange(0, 1))

def randomRotate(im:"ndarray") -> "ndarray":
    """Gaussian filter for bluring the image with random variance.

    Parameters
    ----------
    im :
        

    Returns
    -------

    
    """
    return rotate(im, random.choice([0,90,180,270]), axes=(2,1), reshape=False)

def randomFilter(im:"ndarray") -> "ndarray":
    """randomly selects an exposure filter from histogram equalizers, contrast adjustments, and intensity rescaler and applys it on the input image.
    filters include: equalize_adapthist, equalize_hist, rescale_intensity, adjust_gamma, adjust_log, adjust_sigmoid, gaussian

    Parameters
    ----------
    im :
        

    Returns
    -------

    
    """
    Filters = [randomGamma, randomGaussian, randomIntensity] #equalize_adapthist, equalize_hist, adjust_log, adjust_sigmoid, 
    filt = random.choice(Filters)
    return filt(im)


def randomNoise(im:"ndarray") -> "ndarray":
    """Random gaussian noise with random variance.

    Parameters
    ----------
    im :
        

    Returns
    -------

    
    """
    var = randRange(0.001, 0.01)
    return random_noise(im, var=var)#im + np.random.normal(0, var, 1)#random_noise(im, var=var)

def augment(im:"ndarray", Steps:list=[randomGamma]) -> "ndarray": #randomCrop #randomAffine, ,randomAffine,randomRotate
    """Image augmentation by doing a series of transformations on the image.

    Parameters
    ----------
    im :
        
    Steps :
        (Default value = [randomGamma])

    Returns
    -------

    
    """
    #im = random.choice([randomFilter, randomNoise])(im)
    for step in Steps:
        #if int(randRange(0, len(Steps))):
        im = step(im)
    
    #print(im.shape)
    return im