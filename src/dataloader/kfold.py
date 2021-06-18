from sklearn.model_selection import StratifiedKFold as sk_StratifiedKFold
import numpy as np
import torch

class KFold(sk_StratifiedKFold):
    """Custom KFold based on the sklearn KFold adapted for pytorch dataloader"""
    __doc__ += sk_StratifiedKFold.__doc__
    
    def __init__(self, n_splits:int=5, shuffle:bool=False, random_state:int=None):
        """

        Parameters
        ----------
        n_splits :
            (Default value = 5)
        shuffle :
            (Default value = False)
        random_state :
            (Default value = None)

        Returns
        -------

        
        """
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        
        self._folds = []
        self.fold_idx = 0
        self.subset = None
        
    def split(self,X,y=None):
        """Extended version of the split for a pytorch dataloader's dataset

        Args:
          X: List
          y: (Default value = None)

        Returns:
          type: output (self): Return a self object to reference the next fold.

        Raises:

        """

        self.fold_idx = 0
        self._folds = list(super(sk_StratifiedKFold, self).split(X,y=y))

        return self

    def next(self):
        """ """
        if self.fold_idx < self.n_splits:
            self.fold_idx += 1
            return self.data()
        return False
        
    
    def data(self) -> None:
        """Access the data fold in dataset

        Args:

        Returns:

        Raises:

        """

        return self._folds[self.fold_idx]