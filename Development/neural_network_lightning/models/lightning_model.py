import pytorch_lightning as pl
import torch.nn as nn
import torch
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
        
        # log values
        self.logger.experiment.add_scalar('Train/Loss', loss)

        return {'loss': loss}
  
    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        x, y = batch
        logits = self.forward(x) 
        loss = self.loss(logits, y) 
        
        # log values
        self.logger.experiment.add_scalar('Val/Loss', loss)
        return {'val_loss': loss}
    
    def test_step(self, batch: dict, batch_idx: int) -> dict:
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        
        # log values
        self.logger.experiment.add_scalar('Test/Loss', loss)
        
        return {'test_loss': loss}
      
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        #self.log({'avg_val_loss': avg_loss, 'log': tensorboard_logs})
        self.log("val_loss", avg_loss)

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        #tensorboard_logs = {'test_loss': avg_loss}
        self.log("test_loss", avg_loss)
        #self.log({'avg_test_loss': avg_loss, 'log': tensorboard_logs})

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = {'scheduler': torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.95),
                        'name': 'expo_lr'}
        return [optimizer], [lr_scheduler]
