import matplotlib.pyplot as plt
import torchvision
import numpy as np
from skimage.transform import resize
import torch
from . import utils

__all__ = ['plot_CAM_grid','interactive_slices','interactive_slices_masked']

@utils.figure_decorator
def plot_CAM_grid(image, mask, layer=None, predicted_label=None, expected_label=None, resize_shape=(79,224,224), cmap='jet', alpha=0.3,extractor=None, predicted_override=False, fig=None):
    assert len(image.shape) == 3 and len(mask.shape) == 3
    assert isinstance(image, np.ndarray) and isinstance(mask, np.ndarray)
    
    mask = resize(mask,resize_shape)
    # Expects an image of shape (N,C,D,H,W) and a mask of shape (N,D,C,H,W)
    # Do normalization and stuff
    plt_image = torch.from_numpy((image * 255).astype(np.uint8)).unsqueeze(1)
    plt_mask = torch.from_numpy((mask * 255).astype(np.uint8)).unsqueeze(1)
    
    assert plt_image.shape == (resize_shape[0], 1, *resize_shape[1:]) and plt_mask.shape == (resize_shape[0], 1, *resize_shape[1:])

    # Convert to grid
    grid_img = torchvision.utils.make_grid(plt_image, nrow=10)[0]
    grid_mask = torchvision.utils.make_grid(plt_mask, nrow=10)[0]

    # Remove color and replace it with cmap!
    #fig = plt.figure(figsize=(20,20))
    plt.imshow(grid_img,**{'cmap':'gray'}) 
    im = plt.imshow(grid_mask,**{'cmap':cmap, 'alpha':alpha}) 
    plt.title(f"Layar()er: '{layer}', Extractor: '{extractor}', Predicted: '{predicted_label}', Expected: {expected_label}, Predicted overrided: '{predicted_override}'")
    plt.xlabel('Slices')
    plt.ylabel('Slices')
    fig.colorbar(im, shrink=0.65)

    return fig

class interactive_slices:
    # In order to use this enable %matplotlib widget return with %matplotlib inline
    def __init__(self):
        self.ax=None
        self.fig=None
    def remove_keymap_conflicts(self,new_keys_set):
        for prop in plt.rcParams:
            if prop.startswith('keymap.'):
                keys = plt.rcParams[prop]
                remove_list = set(keys) & new_keys_set
                for key in remove_list:
                    keys.remove(key)
    
    def multi_slice_viewer(self,volume):
        self.remove_keymap_conflicts({'j', 'k'})
        fig, ax = plt.subplots()
        ax.volume = volume
        ax.index = volume.shape[0] // 2
        ax.imshow(volume[ax.index])
        fig.canvas.mpl_connect('key_press_event', self.process_key)
        self.ax = ax
        self.fig=fig
        self.draw()

    def process_key(self,event):
        fig = event.canvas.figure
        ax = fig.axes[0]
        if event.key == 'j':
            self.previous_slice(ax)
        elif event.key == 'k':
            self.next_slice(ax)
        
        fig.canvas.draw()
        self.draw()

    def previous_slice(self,ax):
        volume = ax.volume
        ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
        ax.images[0].set_array(volume[ax.index])

    def next_slice(self,ax):
        volume = ax.volume
        ax.index = (ax.index + 1) % volume.shape[0]
        ax.images[0].set_array(volume[ax.index])
        
    def draw(self):
        self.ax.set_title(f'Layer: {self.ax.index}')
        self.fig.canvas.draw()
        
    def close(self):
        plt.close(self.fig)
        
class interactive_slices_masked:
    # In order to use this enable %matplotlib widget return with %matplotlib inline
    def __init__(self):
        self.ax=None
        self.fig=None

    def remove_keymap_conflicts(self,new_keys_set):
        for prop in plt.rcParams:
            if prop.startswith('keymap.'):
                keys = plt.rcParams[prop]
                remove_list = set(keys) & new_keys_set
                for key in remove_list:
                    keys.remove(key)

    def multi_slice_viewer(self, image, mask, resize_shape=(79,224,224)):
        # Expects an input image: torch.Size([79, 224, 224]) and an activation map: (79, 224, 224)
        assert tuple(image.shape) == resize_shape, f"Image is wrong shape {image.shape}"
        assert len(mask.shape) == len(resize_shape), f"Mask is wrong length {len(mask.shape)}"
        self.remove_keymap_conflicts({'j', 'k'})
        
        mask = resize(mask, resize_shape)
    
        fig, ax = plt.subplots()
        ax.image = self.asType(image)
        ax.mask = self.asType(mask)
        ax.index = ax.image.shape[0] // 2
        ax.imshow(ax.image[ax.index],**{'cmap':'gray'})
        ax.imshow(ax.mask[ax.index],**{'cmap':'jet', 'alpha':0.3})
        fig.canvas.mpl_connect('key_press_event', self.process_key)
        self.ax = ax
        self.fig=fig
        
        self.draw()
        print("Got here!")

    def process_key(self,event):
        print("Got here212!")
        if event.key == 'j':
            self.update(direction=-1)
        elif event.key == 'k':
            self.update(direction=1)
    
    def update(self, direction=1):
        print("Got here!")
        #ax = self.ax
        self.ax.index = (self.ax.index + direction) % self.ax.image.shape[0]
        self.ax.images[0].set_array(self.ax.image[self.ax.index])
        self.ax.images[1].set_array(self.ax.mask[self.ax.index])
        
        self.draw()
        
    def draw(self):
        self.ax.set_title(f'Layer: {self.ax.index}')
        self.fig.canvas.draw()
    
    def asType(self, array):
        return (array * 255).astype(np.uint8)
    
    def cycle(self, timer):
        import time
        for _ in range(self.ax.index):
            self.update()
            time.sleep(timer)