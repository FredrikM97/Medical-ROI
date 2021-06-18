import pytorch_lightning as pl

__all__ = ['DebugCallback']

class DebugCallback(pl.callbacks.Callback):
    """ """
    def __init__(self):
        """ """
        super().__init__()

    def on_fit_start(self, trainer, pl_module) -> None:
        """Called when fit begins

        Args:
          trainer: 
          pl_module: 

        Returns:

        Raises:

        """
        print("Fit begins")


    def on_fit_end(self, trainer, pl_module) -> None:
        """Called when fit ends

        Args:
          trainer: 
          pl_module: 

        Returns:

        Raises:

        """
        print("Fit ends")


    def on_sanity_check_start(self, trainer, pl_module) -> None:
        """Called when the validation sanity check starts.

        Args:
          trainer: 
          pl_module: 

        Returns:

        Raises:

        """
        print("Validation sanity check starts")


    def on_sanity_check_end(self, trainer, pl_module) -> None:
        """Called when the validation sanity check ends.

        Args:
          trainer: 
          pl_module: 

        Returns:

        Raises:

        """
        print("Validation sanity check ends")


    def on_train_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx
    ) -> None:
        """Called when the train batch begins.

        Args:
          trainer: 
          pl_module: 
          batch: 
          batch_idx: 
          dataloader_idx: 

        Returns:

        Raises:

        """
        print("train batch begins")


    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ) -> None:
        """Called when the train batch ends.

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
        print("train batch ends")


    def on_train_epoch_start(self, trainer, pl_module) -> None:
        """Called when the train epoch begins.

        Args:
          trainer: 
          pl_module: 

        Returns:

        Raises:

        """
        print("train epoch begins")


    def on_train_epoch_end(self, trainer, pl_module, outputs) -> None:
        """Called when the train epoch ends.

        Args:
          trainer: 
          pl_module: 
          outputs: 

        Returns:

        Raises:

        """
        print("train epoch ends")


    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        """Called when the val epoch begins.

        Args:
          trainer: 
          pl_module: 

        Returns:

        Raises:

        """
        print("val epoch begins")


    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        """Called when the val epoch ends.

        Args:
          trainer: 
          pl_module: 

        Returns:

        Raises:

        """
        print("val epoch ends")


    def on_test_epoch_start(self, trainer, pl_module) -> None:
        """Called when the test epoch begins.

        Args:
          trainer: 
          pl_module: 

        Returns:

        Raises:

        """
        print("test epoch begins")


    def on_test_epoch_end(self, trainer, pl_module) -> None:
        """Called when the test epoch ends.

        Args:
          trainer: 
          pl_module: 

        Returns:

        Raises:

        """
        print("test epoch ends")


    def on_epoch_start(self, trainer, pl_module) -> None:
        """Called when the epoch begins.

        Args:
          trainer: 
          pl_module: 

        Returns:

        Raises:

        """
        print("epoch begins")


    def on_epoch_end(self, trainer, pl_module) -> None:
        """Called when the epoch ends.

        Args:
          trainer: 
          pl_module: 

        Returns:

        Raises:

        """
        print("epoch ends")


    def on_batch_start(self, trainer, pl_module) -> None:
        """Called when the training batch begins.

        Args:
          trainer: 
          pl_module: 

        Returns:

        Raises:

        """
        print("training batch begins")


    def on_batch_end(self, trainer, pl_module) -> None:
        """Called when the training batch ends.

        Args:
          trainer: 
          pl_module: 

        Returns:

        Raises:

        """
        print("training batch ends")


    def on_train_start(self, trainer, pl_module) -> None:
        """Called when the train begins.

        Args:
          trainer: 
          pl_module: 

        Returns:

        Raises:

        """
        print("train begins")


    def on_train_end(self, trainer, pl_module) -> None:
        """Called when the train ends.

        Args:
          trainer: 
          pl_module: 

        Returns:

        Raises:

        """
        print("train ends")
    

    def on_validation_start(self, trainer, pl_module) -> None:
        """Called when the validation loop begins.

        Args:
          trainer: 
          pl_module: 

        Returns:

        Raises:

        """
        print("validation loop begins")


    def on_validation_end(self, trainer, pl_module) -> None:
        """Called when the validation loop ends.

        Args:
          trainer: 
          pl_module: 

        Returns:

        Raises:

        """
        print("validation loop ends")


    def on_test_start(self, trainer, pl_module) -> None:
        """Called when the test begins.

        Args:
          trainer: 
          pl_module: 

        Returns:

        Raises:

        """
        print("test begins")


    def on_test_end(self, trainer, pl_module) -> None:
        """Called when the test ends.

        Args:
          trainer: 
          pl_module: 

        Returns:

        Raises:

        """
        print("test ends")
