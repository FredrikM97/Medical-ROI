from architectures import create_architecture #testModel

import pytorch_lightning as pl
import torch.nn as nn
import torch
from pytorch_lightning.metrics import functional as FM
from pytorch_lightning.metrics import Accuracy

# https://pytorch-lightning.readthedocs.io/en/latest/metrics.html
class LightningModel(pl.LightningModule): 
    def __init__(self, **hparams):
        super().__init__() 
        self.save_hyperparameters()
        
        self.model = create_architecture(architecture=self.hparams.architecture,input_channels=1, num_classes=3)#testModel(input_channels=1, num_classes=3)
        self.loss = nn.CrossEntropyLoss() 
        self.lr = self.hparams.lr
        
        # These have internal memory
        self.train_metric = Accuracy(compute_on_step=False)
        self.val_metric = Accuracy(compute_on_step=False)
        
        
        
        self.save_hyperparameters()
        
    def training_step(self, batch: dict, batch_idx: int) -> dict:
        x, target = batch
        pred = self.model(x) 
        loss = self.loss(pred, target) 
        print("target",target,"predict",pred)
        self.train_metric(pred,target)
        self.logger.experiment.add_scalars(f"prediction_vs_target/train",{'predict':pred, 'target':target})
        
        return {'loss':loss, 'predict':pred, 'target':target}
  
    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        x, target = batch
        pred = self.model(x) 
        loss = self.loss(pred, target) 
        
        self.val_metric(pred,target)
        
        return {'loss':loss, 'predict':pred, 'target':target}
    
    def on_epoch_end(self):
        self.trainer.datamodule._on_epoch_end()
       
    
    def training_epoch_end(self, outputs):
        # logging histograms
        self.custom_histogram_adder()
        
        loss = avg_metric(outputs, 'loss')
        acc = self.train_metric.compute()
        
        self._add_scalar('train', loss, acc)
    
    def validation_epoch_end(self, outputs):
        loss = avg_metric(outputs, 'loss')
        acc = self.val_metric.compute()
        
        self._add_scalar('val', loss, acc)
        
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return [optimizer]
    
    def custom_histogram_adder(self):
        # iterating through all parameters
        for name,params in self.model.named_parameters():
            self.logger.experiment.add_histogram(name,params,self.current_epoch)
    
    def _add_scalar(self, prefix, loss, acc):
        self.logger.experiment.add_scalar(f"loss/"+prefix,
                                            loss,
                                            self.current_epoch)
        
        self.logger.experiment.add_scalar(f"accuracy/"+prefix,
                                            acc,
                                            self.current_epoch)
def avg_metric(outputs, metric):
    return torch.stack([x[metric] for x in outputs]).mean()

