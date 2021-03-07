#from neural_network.utils.hooks import ActivationMapHook
from src.utils import plot
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import torch

__all__ = []

class ActivationMapCallback(pl.callbacks.Callback):
    
    def __init__(self, model):
        self.hooks = []
        for name, module in model.named_modules():
            if type(module) == torch.nn.modules.conv.Conv3d:
                self.hooks.append(utils.hooks.ActivationMapHook(module, name))
                
    @plot.figure_decorator  
    def on_epoch_end(self, trainer, pl_module, fig=None):
        print("Activation map!")
        set_to_train = False
        if trainer.model.training:
            set_to_train = True
        

        trainer.model.eval()

        for hook in self.hooks:
            hook.register()
        
 
        
        for i, sample in enumerate(pl_module.val_dataloader()):   # Stepping through dataloader might mess up validation elsewhere ?
            # Dont call cuda directly (this is not optimized :( ))
            trainer.model((sample[0][0, np.newaxis, :, :, :, :].cuda(),sample[1][0, np.newaxis].cuda()), i) #trainer.model(sample[0][0, np.newaxis, :, :, :, :].cuda())
            break
 
              
        for hook in self.hooks:
            
            hard_filter_limit = 100
            filters_to_use = min(hook.features.shape[1], hard_filter_limit)
            
            width = 10
            height = int(np.ceil(filters_to_use / width))
            size = 1
            slice_index = int(hook.features.shape[4] / 2)
            min_value = 0
            max_value = 1
            
            # TODO: Dont create a new image at each hook.. Or split this function into two and call figure_decorator
            fig, axs = plt.subplots(height, width, figsize = (width * size, height * size))
            plt.subplots_adjust(wspace = 0, hspace = 0)
            fig.suptitle(f"{hook.name} (Showing {filters_to_use}/{hook.features.shape[1]} filters) (slice [:, :{slice_index}]) size [{hook.features.shape[2]}, {hook.features.shape[3]}])")
            
            for i in range(width * height):
                ax = axs[int(i / width), i % width]
                if i < filters_to_use:
                    colorplot = ax.imshow(hook.features[0][i][:, :, slice_index], vmin = min_value, vmax = max_value,alpha=0.5, cmap='jet')#cmap = 'viridis')
                    ax.set_xticks([])
                    ax.set_yticks([])
                else:
                    ax.set_visible(False)
                    
            fig.subplots_adjust(right = 0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            fig.colorbar(colorplot, cax = cbar_ax)

            trainer.logger.experiment.add_figure("featuremap",fig,trainer.current_epoch) 
            #fig.show()
            #plt.close()
            # https://www.tensorflow.org/tensorboard/image_summaries
            # Save to buffer, write to tb
            
        for hook in self.hooks:
            hook.unregister()
            
        if set_to_train:
            trainer.model.train()