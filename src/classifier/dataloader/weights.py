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