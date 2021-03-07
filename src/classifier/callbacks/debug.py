import pytorch_lightning as pl

__all__ = ['DebugCallback']

class DebugCallback(pl.callbacks.Callback):
    def __init__(self):
        super().__init__()

    def on_fit_start(self, trainer, pl_module) -> None:
        """Called when fit begins"""
        print("Fit begins")


    def on_fit_end(self, trainer, pl_module) -> None:
        """Called when fit ends"""
        print("Fit ends")


    def on_sanity_check_start(self, trainer, pl_module) -> None:
        """Called when the validation sanity check starts."""
        print("Validation sanity check starts")


    def on_sanity_check_end(self, trainer, pl_module) -> None:
        """Called when the validation sanity check ends."""
        print("Validation sanity check ends")


    def on_train_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx
    ) -> None:
        """Called when the train batch begins."""
        print("train batch begins")


    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ) -> None:
        """Called when the train batch ends."""
        print("train batch ends")


    def on_train_epoch_start(self, trainer, pl_module) -> None:
        """Called when the train epoch begins."""
        print("train epoch begins")


    def on_train_epoch_end(self, trainer, pl_module, outputs) -> None:
        """Called when the train epoch ends."""
        print("train epoch ends")


    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        """Called when the val epoch begins."""
        print("val epoch begins")


    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        """Called when the val epoch ends."""
        print("val epoch ends")


    def on_test_epoch_start(self, trainer, pl_module) -> None:
        """Called when the test epoch begins."""
        print("test epoch begins")


    def on_test_epoch_end(self, trainer, pl_module) -> None:
        """Called when the test epoch ends."""
        print("test epoch ends")


    def on_epoch_start(self, trainer, pl_module) -> None:
        """Called when the epoch begins."""
        print("epoch begins")


    def on_epoch_end(self, trainer, pl_module) -> None:
        """Called when the epoch ends."""
        print("epoch ends")


    def on_batch_start(self, trainer, pl_module) -> None:
        """Called when the training batch begins."""
        print("training batch begins")


    def on_batch_end(self, trainer, pl_module) -> None:
        """Called when the training batch ends."""
        print("training batch ends")


    def on_train_start(self, trainer, pl_module) -> None:
        """Called when the train begins."""
        print("train begins")


    def on_train_end(self, trainer, pl_module) -> None:
        """Called when the train ends."""
        print("train ends")
    

    def on_validation_start(self, trainer, pl_module) -> None:
        """Called when the validation loop begins."""
        print("validation loop begins")


    def on_validation_end(self, trainer, pl_module) -> None:
        """Called when the validation loop ends."""
        print("validation loop ends")


    def on_test_start(self, trainer, pl_module) -> None:
        """Called when the test begins."""
        print("test begins")


    def on_test_end(self, trainer, pl_module) -> None:
        """Called when the test ends."""
        print("test ends")
