import pytorch_lightning as pl
import torch.nn as nn
import torch
from pytorch_lightning.metrics import functional as FM
from models.architectures import testModel

class LightningModel(testModel): 
    def __init__(self, hparams):
        
        if hparams is None:
            hparams = {}
        super().__init__(input_channels=1, num_classes=3) #hparams
        
        self.loss = nn.CrossEntropyLoss() 
        self.lr = hparams.get('lr')
        self.save_hyperparameters()

    def training_step(self, batch: dict, batch_idx: int) -> dict:
        x, y = batch
        logits = self.forward(x) 
        loss = self.loss(logits, y) 
        acc = FM.accuracy(logits, y)
        
        # log values
        #self.logger.experiment.add_scalar('Train/Loss', loss)

        self.log_dict( {'train/acc': acc, 'train/loss': loss})
        return  {'acc': acc, 'loss': loss}
  
    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        x, y = batch
        logits = self.forward(x) 
        loss = self.loss(logits, y) 
        acc = FM.accuracy(logits, y)
        
        # log values
        #self.logger.experiment.add_scalar('Val/Loss', loss)
        metrics = {'val/acc': acc, 'val/loss': loss}
        self.log_dict(metrics)
        return metrics
    
    def test_step(self, batch: dict, batch_idx: int) -> dict:
        metrics = self.validation_step(batch, batch_idx)
        metrics = {'test/acc': metrics['val/acc'], 'test/loss': metrics['val/loss']}
        self.log_dict(metrics)
        return metrics
        
    def validation_epoch_end(self, outputs):

        pass
    def test_epoch_end(self, outputs):

        pass
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        return [optimizer]
