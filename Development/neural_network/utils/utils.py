import torch
import pytorch_lightning as pl
import os
import matplotlib.pyplot as plt

def label_encoder(labels):
    "Convert list of string labels to tensors"
    from sklearn import preprocessing
    import torch
    le = preprocessing.LabelEncoder()
    targets = le.fit_transform(labels)
                  
    return torch.as_tensor(targets)

def get_availible_files(path, contains:str=''):
    return [f for f in os.listdir(path) if contains in f]

class meanMetric(pl.metrics.Metric):
    def __init__(self,compute_on_step=False):
        super().__init__(compute_on_step=compute_on_step, dist_sync_on_step=False)
        self.add_state('val',default=torch.tensor(0.), dist_reduce_fx='mean')
        self.add_state('num',default=torch.tensor(0.), dist_reduce_fx='mean')
    def update(self, loss):
        self.val+=loss.mean()
        self.num +=1
        
    def compute(self):
        return self.val/self.num
    
def ROC(trainer, roc_classes, prefix=''):
    fig = plt.figure(figsize = (10,7))
    lw = 2
    colors = ['aqua', 'darkorange', 'cornflowerblue']
    fpr, tpr, threshold = roc_classes
    for i,(_,color) in enumerate(zip(roc_classes,colors)):
        
        _fpr = fpr[i]
        _tpr = tpr[i]
        _threshold = threshold[i]
        
        #threshold = threshold#.detach()
        #print("ASdasdasd", fpr, tpr, threshold)
        #

        

        #self.logger.experiment.add_text(f"sensitivity/{phase}", str(tpr))
        #self.logger.experiment.add_text(f"specificity/{phase}", str(1-fpr))
        try:
            auc = pl.metrics.functional.auc(_fpr,_tpr).cpu().numpy()
            
            _fpr = _fpr.cpu().numpy()
            _tpr = _tpr.cpu().numpy()
            plt.plot(_fpr, _tpr, color=color, lw=lw,
                     label='ROC curve of class {0} (area={1:0.2f} tpr={2:0.2f} fpr={3:0.2f})'
                     ''.format(i,auc, _tpr.mean(), 1-_fpr.mean())) #
        except Exception as e:
            print(e)
            
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.close()

    trainer.logger.experiment.add_figure(f"ROC/{prefix}", fig, trainer.current_epoch)