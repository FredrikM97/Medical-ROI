from scipy import ndimage
import numpy as np
from skimage import exposure, filters, measure

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