from sklearn.model_selection import KFold
class Kfold:
    def __init__(self):
        self._fold_idx = None
        self._n_splits = None
        self._shuffle=None
        self._data = None
        self._random_state = None

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
