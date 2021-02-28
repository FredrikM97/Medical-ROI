import pytorch_lightning as pl
import utils 
import matplotlib.pyplot as plt
import seaborn as sns
import torch

class MetricCallback(pl.callbacks.Callback):
    def __init__(self, num_classes=3):
        super().__init__()
        self.num_classes = 3
        self.logger_struct = lambda metric_prefix, prefix, metric: {f"{metric_prefix}/{prefix}": metric}
        self.val_metrics = utils.storeMetrics().cuda()
            
    def on_validation_batch_end(self,trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.val_metrics(
            outputs['predicted/val'],
            outputs['target/val'],
            outputs['probability/val'],
            outputs['loss/val']
        )

    def on_train_epoch_end(self,trainer, pl_module, outputs):
        avg_loss = torch.stack([x[0]['minimize'] for x in outputs[0]]).mean()
        self.add_scalar(trainer, pl_module, avg_loss, metric_prefix='loss',prefix='train')

    def on_validation_epoch_end(self,trainer, pl_module):
        pred, target, prob, loss = self.val_metrics.compute()
        
        # Log data from validation
        self.custom_histogram_adder(trainer)
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
        self.add_scalar(
            trainer, 
            pl_module,
            pl.metrics.functional.accuracy(pred, target),
            metric_prefix='accuracy',
            prefix='val'
        )
        self.add_scalar(
            trainer,
            pl_module,
            loss.mean(),
            metric_prefix='loss',
            prefix='val'
        )

        # Reset the states.. we dont want to keep it in memory! As of pytorch-lightning 1.2 this is not done automatically!
        self.val_metrics.reset()
    
    def cm_plot(self, trainer, cm, prefix=''):
     
        fig=plt.figure();
        ax = sns.heatmap(utils.to_cpu_numpy(cm), annot=True, annot_kws={"size": 12})
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        trainer.logger.experiment.add_figure(f"confmat/{prefix}", fig,trainer.current_epoch)
    
    def roc_plot(self, trainer, roc_classes, prefix=''):
        (auc, fpr, tpr), roc_fig = utils.ROC(roc_classes)
        
        trainer.logger.log_metrics(
            {
                f"auc/{prefix}": auc,
                f"specificity/{prefix}":1-fpr,
                f"sensitivity/{prefix}":tpr
            },step=trainer.current_epoch
        ) 
        trainer.logger.experiment.add_figure(f"ROC/{prefix}", roc_fig, trainer.current_epoch)
        
    def add_scalar(self, trainer, pl_module, metric, metric_prefix=None,prefix=''):
        assert metric_prefix
        trainer.logger.log_metrics(self.logger_struct(metric_prefix, prefix, metric), step=trainer.current_epoch)
        
    def custom_histogram_adder(self, trainer):
        # iterating through all parameters
        for name,params in trainer.model.named_parameters():
            trainer.logger.experiment.add_histogram(name,params,trainer.current_epoch)
            