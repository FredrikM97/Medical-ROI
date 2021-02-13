from hooks import ActivationMapHook

from pytorch_lightning.callbacks import Callback
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class ActivationMap(Callback):
    
    def __init__(self, model):
        self.hooks = []
        for name, module in model.named_modules():
            if type(module) == torch.nn.modules.conv.Conv3d:
                self.hooks.append(ActivationMapHook(module, name))
    
    def on_epoch_end(self, trainer, pl_module):
        
        set_to_train = False
        if trainer.model.training:
            set_to_train = True
            
        trainer.model.eval()
        
        for hook in self.hooks:
            hook.register()
            
        for i, sample in enumerate(pl_module.val_dataloader()):   # Stepping through dataloader might mess up validation elsewhere ?
            trainer.model(sample[0][0, np.newaxis, :, :, :, :].cuda())
            break
            
        for hook in self.hooks:
            
            hard_filter_limit = 100
            filters_to_use = min(hook.features.shape[1], hard_filter_limit)
            
            width = 10
            height = int(np.ceil(filters_to_use / width))
            size = 1
            slice_index = int(hook.features.shape[4] / 2)
            min_value = hook.features[0].min()
            max_value = hook.features[0].max()
            
            fig, axs = plt.subplots(height, width, figsize = (width * size, height * size))
            plt.subplots_adjust(wspace = 0, hspace = 0)
            fig.suptitle(hook.name + ' (Showing ' + str(filters_to_use) + '/' + str(hook.features.shape[1]) + ' filters)' + ' (slice [:, : ' + str(slice_index) + '])' + ' (size [' + str(hook.features.shape[2]) + ', ' + str(hook.features.shape[3]) + '])' )
            
            for i in range(width * height):
                ax = axs[int(i / width), i % width]
                if i < filters_to_use:
                    colorplot = ax.imshow(hook.features[0][i][:, :, slice_index], vmin = min_value, vmax = max_value, cmap = 'viridis')
                    ax.set_xticks([])
                    ax.set_yticks([])
                else:
                    ax.set_visible(False)
                    
            fig.subplots_adjust(right = 0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            fig.colorbar(colorplot, cax = cbar_ax)
            
            trainer.logger.experiment.add_figure("featuremap",fig,trainer.current_epoch) 
            #fig.show()
            
            # https://www.tensorflow.org/tensorboard/image_summaries
            # Save to buffer, write to tb
            
        for hook in self.hooks:
            hook.unregister()
            
        if set_to_train:
            trainer.model.train()
        
            
class ConfusionMatrix(Callback):
    # TODO: Add so model store predicted and target. Then we can reuse that instead of external stuff..
    def on_epoch_end(self, trainer, pl_module):
     
        fig=plt.figure();
        cm = trainer.model.train_cm.compute()
        ax = sns.heatmap(cm.detach().cpu().numpy(), annot=True, annot_kws={"size": 12})
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        trainer.logger.experiment.add_figure("confmat/train", fig,trainer.current_epoch)