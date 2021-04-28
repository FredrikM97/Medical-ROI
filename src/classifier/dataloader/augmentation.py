# Based on https://www.kaggle.com/safavieh/image-augmentation-using-skimage

from skimage.transform import warp, AffineTransform, ProjectiveTransform
from skimage.exposure import equalize_adapthist, equalize_hist, rescale_intensity, adjust_gamma, adjust_log, adjust_sigmoid
from skimage.filters import gaussian
from skimage.util import random_noise
import random
from torchvision import transforms
def randRange(a, b):
    '''
    Generate random float values in desired range
    '''
    return pl.rand() * (b - a) + a


def randomAffine(im):
    '''
    Affine transformation with random scale, rotation, shear and translation parameters
    '''
    tform = AffineTransform(scale=(randRange(0.75, 1.3), randRange(0.75, 1.3)),
                            rotation=randRange(-0.25, 0.25),
                            shear=randRange(-0.2, 0.2),
                            translation=(randRange(-im.shape[1]//10, im.shape[1]//10), 
                                         randRange(-im.shape[2]//10, im.shape[2]//10)))
    return warp(im, tform.inverse, mode='reflect')

def randomCrop(im):
    '''
    croping the image in the center from a random margin from the borders
    '''
    margin = 1/10
    start = [int(randRange(0, im.shape[1] * margin)),
             int(randRange(0, im.shape[2] * margin))]
    end = [int(randRange(im.shape[1] * (1-margin), im.shape[1])), 
           int(randRange(im.shape[2] * (1-margin), im.shape[2]))]
    return im[start[0]:end[0], start[1]:end[1]]


def randomIntensity(im):
    '''
    Rescales the intensity of the image to random interval of image intensity distribution
    '''
    return rescale_intensity(im,
                             in_range=tuple(pl.percentile(im, (randRange(0,10), randRange(90,100)))),
                             out_range=tuple(pl.percentile(im, (randRange(0,10), randRange(90,100)))))

def randomGamma(im):
    '''
    Gamma filter for contrast adjustment with random gamma value.
    '''
    return adjust_gamma(im, gamma=randRange(0.5, 1.5))

def randomGaussian(im):
    '''
    Gaussian filter for bluring the image with random variance.
    '''
    return gaussian(im, sigma=randRange(0, 5))
    
def randomFilter(im):
    '''
    randomly selects an exposure filter from histogram equalizers, contrast adjustments, and intensity rescaler and applys it on the input image.
    filters include: equalize_adapthist, equalize_hist, rescale_intensity, adjust_gamma, adjust_log, adjust_sigmoid, gaussian
    '''
    Filters = [equalize_adapthist, equalize_hist, adjust_log, adjust_sigmoid, randomGamma, randomGaussian, randomIntensity]
    filt = random.choice(Filters)
    return filt(im)


def randomNoise(im):
    '''
    Random gaussian noise with random variance.
    '''
    var = randRange(0.001, 0.01)
    return random_noise(im, var=var)

def augment(im, Steps=[randomAffine, randomFilter, randomNoise, randomCrop]):
    '''
    Image augmentation by doing a series of transformations on the image.
    '''
    for step in Steps:
        im = step(im)
    return im