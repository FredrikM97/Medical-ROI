from sklearn.model_selection import KFold
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import os
import numpy as np

__all__ = ['ClassWeights','Kfold','split_data','get_labels','get_nii_files']

class ClassWeights:
    def __init__(self, classes:dict, delimiter:str):
        self._classes=classes
        self._delimiter=delimiter
        self._weights = None
        
    @property
    def weights(self):
        return self._weights
    
    @weights.setter
    def weights(self, images):
        self._labels = get_labels(images, self._classes, self._delimiter)
        
        # Compute weights
        self._weights = torch.from_numpy(sklearn.utils.class_weight.compute_class_weight('balanced', classes=list(self._classes.keys()), y=labels)).float().cuda()
    
    def numpy(self):
        return self._weights.numpy()
    
class Kfold:
    
    def __init__(self):
        self._fold_idx = None
        self._n_splits = None
        self._shuffle=None
        self._data = None
        self._random_state = None
        #self._kfold(dataset, n_splits, shuffle=shuffle, random_state=random_state)
        
    def kfold(self,dataset, n_splits, shuffle=False, random_state=None):
        self._shuffle = shuffle
        self._n_splits = n_splits
        self._random_state = random_state
        idxs = KFold(n_splits, shuffle=shuffle, random_state=random_state).split(np.arange(len(dataset)))
        self._folds(((torch.utils.data.Subset(dataset, train_idxs), torch.utils.data.Subset(dataset, val_idxs)) for train_idxs, val_idxs in idxs))

    def folds(self,folds):
        self._fold_idx = 1
        self._folds = folds
    
    @property 
    def data(self):
        return self._data
    
    @property
    def random_state(self):
        return self._random_state
        
    @property
    def shuffle(self):
        return self._shuffle
    
    @property
    def n_splits(self):
        return self._n_splits
    
    @property
    def fold_idx(self) -> int:
        return self._fold_idx

    def has_folds(self) -> bool:
        assert self.data != None, "No data initiated for kfold!"
        return self.fold_idx < self.n_splits
        
    def next(self) -> None:
        assert self.has_folds(), "Cant find more folds!"
        self._fold_idx += 1
        return next(self._folds)
    
def split_data(data,test_size=0.1, random_state=None, shuffle=False):
    return train_test_split(data, test_size=test_size, random_state=random_state, shuffle=shuffle)

def get_labels(images:list, classes:dict, delimiter:str):
    assert delimiter in images[0], "The defined delimiter for ClassWeights could not be found in image input!"
    return np.array([classes[img_path.rsplit("/",1)[1].split(delimiter,1)[0]] for img_path in images])

def get_nii_files(srcdir):
    return [
            path + '/' + filename
            for path, _, files in os.walk(srcdir) 
            for filename in files if filename.endswith('.nii')
        ] 