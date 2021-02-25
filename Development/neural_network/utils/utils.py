import torch
import pytorch_lightning as pl
import os

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
        self.add_state('mean',default=torch.tensor(0.), dist_reduce_fx='mean')

    def update(self, loss):
        self.mean+=loss.mean()

    def compute(self):
        return self.mean