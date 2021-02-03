from torchvision import transforms as T 
from torch.nn import functional as F
import cv2

def get_transform(config, method=cv2.INTER_LINEAR):
    transform_list = []

    if 'preprocess' in config:
        if 'resize' in config['preprocess']:
            transform_list.append(T.Lambda(lambda img: F.interpolate(img, size=tuple(config['input_size']), mode='nearest')))
            #transform_list.append(T.Resize(config['input_size'], interpolation=2))
        
        if 'pad' in config['preprocess']:
            transform_list.append(T.Lambda(lambda img: pad_to_shape(img, (config['input_channels'],*config['input_size']))))
            
        if 'totensor' in config['preprocess']:
            transform_list.append(T.ToTensor())

    return T.Compose(transform_list)


def pad_to_shape(this, shp):
    """
    Pads this image with zeroes to shp.
    Args:
        this: image tensor to pad
        shp: desired output shape
    Returns:
        Zero-padded tensor of shape shp.
    """
    if len(shp) == 4:
        pad = (0, shp[3] - this.shape[3], 0, shp[2] - this.shape[2], 0, shp[1] - this.shape[1])
    return F.pad(this, pad)