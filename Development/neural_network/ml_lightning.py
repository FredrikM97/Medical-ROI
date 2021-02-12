from configs import load_config
from models import create_model
from datasets import create_dataset
from pytorch_lightning.trainer.states import TrainerState
from pytorch_lightning.callbacks import ModelCheckpoint, progress, Callback
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import sys
import torch
import numpy as np

class Agent:
    def __init__(self, config_name:str, export:bool=True):
        print('Setup configurations...')
        # ****** Setup configurations ******
        self.config = load_config(config_name)
        
        # ****** Setup seed *******
        #np.random.seed(self.config['agent']['seed'])
        
        # ****** Setup dataloader ******
        self.dataset = create_dataset(**self.config['dataset_params'])
        
        # ****** Setup model ******
        self.model = create_model(**self.config['model_params'], class_weights=self.dataset._get_class_weights())
        self.setup_trainer()
        
        
    def setup_trainer(self):
        self.trainer = pl.Trainer(
            max_epochs=self.config['model_params']['max_epochs'], 
            profiler=None, 
            reload_dataloaders_every_epoch=self.config['reload_dataloaders_every_epoch'],
            gpus=self.get_gpu(), 
            logger=self.logger(),
            callbacks=self.callbacks(),
            progress_bar_refresh_rate=self.config['progress_bar_refresh_rate'],
            num_sanity_val_steps=self.config['num_sanity_val_steps'],
            benchmark=True,
            auto_lr_find=True,
            
        )
        
    def logger(self):
        return pl.loggers.TensorBoardLogger(self.config['logs']['tensorboard'], name=self.config['model_params']['model_name'])
     
    def callbacks(self) -> list:
        lit_progress = LitProgressBar()
        activation_map = ActivationMap(self.model)
        checkpoint_callback = ModelCheckpoint(
            save_top_k=0, # disable?
            filename=self.config['logs']['checkpoint']+'{epoch}-{val_loss:.2f}',
            mode='min', 
        )
        return [lit_progress, checkpoint_callback, activation_map]
    
    def get_gpu(self):
        return -1 if torch.cuda.is_available() else None
    
    def fit(self, cv=False) -> None:
        if self.config['agent']['kfold']:
            self.__fit_cv()
        else:
            self.__fit()
    
    def __fit(self):
        self.trainer.fit(
            self.model, 
            datamodule=self.dataset
        )
    
    def __fit_cv(self) -> None:
        print("Fitting with cv")
         
        while self.dataset._has_folds():
            # call fit
            fold_idx = self.dataset._get_fold()
            print(f"Validation on fold: {fold_idx}")
            self.setup_trainer()
            self.__fit()
            if self.trainer._state == TrainerState.INTERRUPTED: break

            # store metrics
            # get next fold
            self.dataset._next_fold()
 
    def get_info():
        pass
        

class LitProgressBar(progress.ProgressBarBase):
#https://pytorch-lightning.readthedocs.io/en/latest/_modules/pytorch_lightning/callbacks/progress.html#ProgressBarBase.on_validation_batch_end
    def __init__(self):
        super().__init__()  # don't forget this :)
        self.enable()

    def disable(self):
        self._enable = False
    
    def enable(self):
        self._enable = True
        
    def on_epoch_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)
        print()
        print("",end="", flush=True)
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        super().on_train_batch_end(trainer, pl_module, outputs,batch, batch_idx, dataloader_idx) 
        
        con = f'Epoch {trainer.current_epoch+1} [{batch_idx+1:.00f}/{self.total_train_batches:.00f}] {self.get_progress_bar_dict(trainer)}'
        
        self._update(con)
        
    def _update(self,con):
        print(con, end="\r", flush=True)
        
    def get_progress_bar_dict(self,trainer):
        tqdm_dict = trainer.progress_bar_dict
        if 'v_num' in tqdm_dict:
            del tqdm_dict['v_num']
        return tqdm_dict
    
class ActivationMapHook():
    
    def __init__(self, module, name):
        self.module = module
        self.name = name
        self.hook = None
        self.features = None
        
    def register(self):
        self.hook = self.module.register_forward_hook(self.callback)
        
    def unregister(self):
        self.hook.remove()
        
    def callback(self, module, input, output):
        self.features = output.cpu().data.numpy()
    
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