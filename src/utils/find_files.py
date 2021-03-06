def find_using_name(basedir,set_name):
    """Import the module "data/[dataset_name]_dataset.py".
    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = BASEDIR + dataset_name
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '')
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, LightningDataModule):
            dataset = cls

    if dataset is None:
        raise NotImplementedError('In {0}.py, there should be a subclass of BaseDataset with class name that matches {1} in lowercase.'.format(dataset_filename, target_dataset_name))

    return dataset
