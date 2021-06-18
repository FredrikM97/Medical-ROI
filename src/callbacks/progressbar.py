from pytorch_lightning.callbacks import progress
__all__ = ['LitProgressBar']

class LitProgressBar(progress.ProgressBarBase):
    """ """
#https://pytorch-lightning.readthedocs.io/en/latest/_modules/pytorch_lightning/callbacks/progress.html#ProgressBarBase.on_validation_batch_end
    def __init__(self):
        """ """
        super().__init__()  # don't forget this :)
        self.enable()

    def disable(self):
        """ """
        self._enable = False
    
    def enable(self):
        """ """
        self._enable = True
        
    def on_epoch_start(self, trainer, pl_module):
        """

        Args:
          trainer: 
          pl_module: 

        Returns:

        Raises:

        """
        super().on_train_start(trainer, pl_module)

        print("",end="", flush=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """

        Args:
          trainer: 
          pl_module: 
          outputs: 
          batch: 
          batch_idx: 
          dataloader_idx: 

        Returns:

        Raises:

        """
        super().on_train_batch_end(trainer, pl_module, outputs,batch, batch_idx, dataloader_idx) 
        
        con = f'Epoch {trainer.current_epoch+1} [{batch_idx+1:.00f}/{self.total_train_batches:.00f}] {self.get_progress_bar_dict(trainer)}'
        
        self._update(con)
        
    def _update(self,con):
        """

        Args:
          con: 

        Returns:

        Raises:

        """

        print(con, end="\r", flush=True)
        
    def get_progress_bar_dict(self,trainer):
        """

        Args:
          trainer: 

        Returns:

        Raises:

        """
        tqdm_dict = trainer.progress_bar_dict
        if 'v_num' in tqdm_dict:
            del tqdm_dict['v_num']
        return tqdm_dict