import pytorch_lightning as pl

__all__ = ['DebugCallback']

class DebugCallback(pl.callbacks.Callback):
    """ """
    def __init__(self):
        """ """
        super().__init__()

    def on_fit_start(self, trainer, pl_module) -> None:
        """Called when fit begins

        Parameters
        ----------
        trainer :
            
        pl_module :
            

        Returns
        -------

        
        """
        print("Fit begins")


    def on_fit_end(self, trainer, pl_module) -> None:
        """Called when fit ends

        Parameters
        ----------
        trainer :
            
        pl_module :
            

        Returns
        -------

        
        """
        print("Fit ends")


    def on_sanity_check_start(self, trainer, pl_module) -> None:
        """Called when the validation sanity check starts.

        Parameters
        ----------
        trainer :
            
        pl_module :
            

        Returns
        -------

        
        """
        print("Validation sanity check starts")


    def on_sanity_check_end(self, trainer, pl_module) -> None:
        """Called when the validation sanity check ends.

        Parameters
        ----------
        trainer :
            
        pl_module :
            

        Returns
        -------

        
        """
        print("Validation sanity check ends")


    def on_train_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx
    ) -> None:
        """Called when the train batch begins.

        Parameters
        ----------
        trainer :
            
        pl_module :
            
        batch :
            
        batch_idx :
            
        dataloader_idx :
            

        Returns
        -------

        
        """
        print("train batch begins")


    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ) -> None:
        """Called when the train batch ends.

        Parameters
        ----------
        trainer :
            
        pl_module :
            
        outputs :
            
        batch :
            
        batch_idx :
            
        dataloader_idx :
            

        Returns
        -------

        
        """
        print("train batch ends")


    def on_train_epoch_start(self, trainer, pl_module) -> None:
        """Called when the train epoch begins.

        Parameters
        ----------
        trainer :
            
        pl_module :
            

        Returns
        -------

        
        """
        print("train epoch begins")


    def on_train_epoch_end(self, trainer, pl_module, outputs) -> None:
        """Called when the train epoch ends.

        Parameters
        ----------
        trainer :
            
        pl_module :
            
        outputs :
            

        Returns
        -------

        
        """
        print("train epoch ends")


    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        """Called when the val epoch begins.

        Parameters
        ----------
        trainer :
            
        pl_module :
            

        Returns
        -------

        
        """
        print("val epoch begins")


    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        """Called when the val epoch ends.

        Parameters
        ----------
        trainer :
            
        pl_module :
            

        Returns
        -------

        
        """
        print("val epoch ends")


    def on_test_epoch_start(self, trainer, pl_module) -> None:
        """Called when the test epoch begins.

        Parameters
        ----------
        trainer :
            
        pl_module :
            

        Returns
        -------

        
        """
        print("test epoch begins")


    def on_test_epoch_end(self, trainer, pl_module) -> None:
        """Called when the test epoch ends.

        Parameters
        ----------
        trainer :
            
        pl_module :
            

        Returns
        -------

        
        """
        print("test epoch ends")


    def on_epoch_start(self, trainer, pl_module) -> None:
        """Called when the epoch begins.

        Parameters
        ----------
        trainer :
            
        pl_module :
            

        Returns
        -------

        
        """
        print("epoch begins")


    def on_epoch_end(self, trainer, pl_module) -> None:
        """Called when the epoch ends.

        Parameters
        ----------
        trainer :
            
        pl_module :
            

        Returns
        -------

        
        """
        print("epoch ends")


    def on_batch_start(self, trainer, pl_module) -> None:
        """Called when the training batch begins.

        Parameters
        ----------
        trainer :
            
        pl_module :
            

        Returns
        -------

        
        """
        print("training batch begins")


    def on_batch_end(self, trainer, pl_module) -> None:
        """Called when the training batch ends.

        Parameters
        ----------
        trainer :
            
        pl_module :
            

        Returns
        -------

        
        """
        print("training batch ends")


    def on_train_start(self, trainer, pl_module) -> None:
        """Called when the train begins.

        Parameters
        ----------
        trainer :
            
        pl_module :
            

        Returns
        -------

        
        """
        print("train begins")


    def on_train_end(self, trainer, pl_module) -> None:
        """Called when the train ends.

        Parameters
        ----------
        trainer :
            
        pl_module :
            

        Returns
        -------

        
        """
        print("train ends")
    

    def on_validation_start(self, trainer, pl_module) -> None:
        """Called when the validation loop begins.

        Parameters
        ----------
        trainer :
            
        pl_module :
            

        Returns
        -------

        
        """
        print("validation loop begins")


    def on_validation_end(self, trainer, pl_module) -> None:
        """Called when the validation loop ends.

        Parameters
        ----------
        trainer :
            
        pl_module :
            

        Returns
        -------

        
        """
        print("validation loop ends")


    def on_test_start(self, trainer, pl_module) -> None:
        """Called when the test begins.

        Parameters
        ----------
        trainer :
            
        pl_module :
            

        Returns
        -------

        
        """
        print("test begins")


    def on_test_end(self, trainer, pl_module) -> None:
        """Called when the test ends.

        Parameters
        ----------
        trainer :
            
        pl_module :
            

        Returns
        -------

        
        """
        print("test ends")
