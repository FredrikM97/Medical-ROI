import os
import matplotlib.pyplot as plt
import numpy as np

def get_availible_files(path, contains:str=''):
    return [f for f in os.listdir(path) if contains in f]
    
def to_cpu_numpy(data):
    # Send to CPU. If computational graph is connected then detach it as well.
    if data.requires_grad:
        print("Disconnect graph!")
        return data.detach().cpu().numpy()
    else:
        return data.cpu().numpy()
    
class interactive_slices:
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
        
class interactive_slices_masked:
    
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

    def multi_slice_viewer(self, image, mask):
        # Expects an input image: torch.Size([1, 79, 224, 224]) and an activation map: (79, 224, 224)
        self.remove_keymap_conflicts({'j', 'k'})
        fig, ax = plt.subplots()
        ax.image = self.image(image)
        ax.mask = self.mask(mask)
        ax.index = ax.image.shape[0] // 2
        ax.imshow(ax.image[ax.index],**{'cmap':'gray'})
        ax.imshow(ax.mask[ax.index],**{'cmap':'jet', 'alpha':0.3})
        
        
        self.ax = ax
        self.fig=fig
        self.fig.canvas.mpl_connect('key_press_event', self.process_key)
        self.draw()
        

    def process_key(self,event):
        if event.key == 'j':
            self.update(direction=-1)
        elif event.key == 'k':
            self.update(direction=1)
    
    def update(self, direction=1):
        ax = self.ax
        ax.index = (ax.index + direction) % ax.image.shape[0]
        ax.images[0].set_array(ax.image[ax.index])
        ax.images[1].set_array(ax.mask[ax.index])
        
        self.draw()
        
    def draw(self):
        self.ax.set_title(f'Layer: {self.ax.index}')
        self.fig.canvas.draw()
        
    def image(self, image):
        return (image.squeeze(0) * 255).numpy().astype(np.uint8)
    
    def mask(self, mask):
        return (mask * 255).astype(np.uint8)

    def cycle(self, timer):
        import time
        for _ in range(self.ax.index):
            self.update()
            time.sleep(timer)
            
def merge_dict(a, b, path=None):
    "merges b into a"
    if path is None: path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge_dict(a[key], b[key], path + [str(key)])
            else:
                a[key] = b[key]
        else:
            a[key] = b[key]
    return a