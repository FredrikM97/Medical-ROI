"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets. Also
    includes some transformation functions.
"""
from abc import ABC, abstractmethod
import cv2
import numpy as np
import torch.utils.data as data
from torchvision import transforms as T 
#from torchvision.transforms import functional as F
from torch.nn import functional as F

class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.
    """

    def __init__(self, configuration):
        """Initialize the class; save the configuration in the class.
        """
        self.configuration = configuration

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, idx):
        """Return a data point (usually data and labels in
            a supervised setting).
        """
        pass

    def pre_epoch_callback(self, epoch):
        """Callback to be called before every epoch.
        """
        pass

    def post_epoch_callback(self, epoch):
        """Callback to be called after every epoch.
        """
        pass


def get_transform(config, method=cv2.INTER_LINEAR):
    transform_list = []

    if 'preprocess' in config:
        if 'resize' in config['preprocess']:
            transform_list.append(T.Lambda(lambda img: F.interpolate(img, size=tuple(config['input_size']), mode='nearest')))
            #transform_list.append(T.Resize(config['input_size'], interpolation=2))
        
        if 'totensor' in config['preprocess']:
            transform_list.append(T.ToTensor())

    return T.Compose(transform_list)
