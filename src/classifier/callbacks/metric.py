import pytorch_lightning as pl
from src.utils import utils
from src.utils.plots import roc,plot
from src.classifier.metric import MetricTracker
import matplotlib.pyplot as plt
import seaborn as sns
import torch

__all__ = ['MetricCallback']

class MetricCallback(pl.callbacks.Callback):
    def __init__(self, num_classes=3):
        super().__init__()
        self.num_classes = 3
        self.logger_struct = lambda metric_prefix, prefix, metric: (f"{metric_prefix}/{prefix}", metric)
<<<<<<< HEAD:src/classifier/callbacks/metrics.py
<<<<<<< HEAD:src/classifier/callbacks/metrics.py
        self.val_metrics = MetricTracker().cuda()
=======
        self.metricsTracker = utils.MetricsTracker().cuda()
>>>>>>> Bug fixes to improve speed and changed some functionality to reduce complexity:Development/neural_network/callbacks/metrics.py
    
=======
        self.metricsTracker = MetricTracker().cuda()  
>>>>>>> Minor bugfixes to run trainer:src/classifier/callbacks/metric.py
    def on_train_start(self, trainer, *args, **kwargs):
        if not trainer.logger:
            raise Exception('Cannot use callback with Trainer that has no logger.')
        
        
    def on_validation_batch_end(self,trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.metricsTracker(
            outputs['predicted/val'],
            outputs['target/val'],
            outputs['probability/val'],
            outputs['loss/val']
        )

    def on_train_epoch_end(self,trainer, pl_module, outputs):
        trainer.logger.experiment.add_scalar(*self.logger_struct('loss', 'train', torch.stack([x[0]['minimize'] for x in outputs[0]]).mean()), trainer.current_epoch)

    def on_validation_epoch_end(self,trainer, pl_module):
        pred, target, prob, loss = self.metricsTracker.compute()
        
        # Log data from validation
        #self.custom_histogram_adder(trainer)
        self.cm_plot(
            trainer, 
            pl.metrics.functional.confusion_matrix(pred, target, num_classes=self.num_classes), 
            prefix='val'
        )
        
        self.roc_plot(
            trainer,
            pl.metrics.functional.roc(prob, target, num_classes=self.num_classes), 
            prefix='val'
        )
 
        trainer.logger.experiment.add_scalar(*self.logger_struct('accuracy', 'val',pl.metrics.functional.accuracy(pred, target)), trainer.current_epoch)
        trainer.logger.experiment.add_scalar(*self.logger_struct('loss', 'val', loss.mean()), trainer.current_epoch)

        # Reset the states.. we dont want to keep it in memory! As of pytorch-lightning 1.2 this is not done automatically!
        self.metricsTracker.reset()
    
    def cm_plot(self, trainer, cm, prefix=''):
        fig = plt.figure(figsize=(20,20))
<<<<<<< HEAD:src/classifier/callbacks/metrics.py
        ax = sns.heatmap(utils.to_cpu_numpy(cm), annot=True, annot_kws={"size": 12})
>>>>>>> Bug fixes to improve speed and changed some functionality to reduce complexity:Development/neural_network/callbacks/metrics.py
=======
        ax = sns.heatmap(utils.tensor2numpy(cm), annot=True, annot_kws={"size": 12})
>>>>>>> Minor bugfixes to run trainer:src/classifier/callbacks/metric.py
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        trainer.logger.experiment.add_figure(f"confmat/{prefix}", fig,trainer.current_epoch)
        
    def roc_plot(self, trainer, roc_classes, prefix=''):
        (auc, fpr, tpr), roc_fig = roc.ROC(roc_classes)
        
        trainer.logger.log_metrics(
            {
                f"auc/{prefix}": auc,
                f"specificity/{prefix}":1-fpr,
                f"sensitivity/{prefix}":tpr
            },step=trainer.current_epoch
        ) 
        trainer.logger.experiment.add_figure(f"ROC/{prefix}", roc_fig, trainer.current_epoch)

    def custom_histogram_adder(self, trainer):
        # iterating through all parameters
        for name,params in trainer.model.named_parameters():
            trainer.logger.experiment.add_histogram(name,params,trainer.current_epoch)
            