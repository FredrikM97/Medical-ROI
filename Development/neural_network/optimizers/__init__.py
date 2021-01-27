import importlib
from optimizers.base_optimizer import BaseOptimizer
from torch.optim import lr_scheduler


def find_optimizer_using_name(optimizer_name):
    """Import the module "models/[model_name]_model.py".
        In the file, the class called DatasetNameModel() will
        be instantiated. It has to be a subclass of BaseModel,
        and it is case-insensitive.
    """
    optimizer_filename = "optimizers." + optimizer_name + "_optimizer"
    modellib = importlib.import_module(optimizer_filename)
    optimizer = None
    target_optimizer_name = optimizer_filename.split('_',1)[0].split('.',1)[1]#.replace('_', '')

    for name, cls in modellib.__dict__.items():
        if name.lower() == target_optimizer_name.lower() and issubclass(cls, BaseOptimizer):
            optimizer = cls

    if optimizer is None:
        print("In %s.py, there should be a subclass of BaseOptimizer with class name that matches %s in lowercase." % (optimizer_filename, target_optimizer_name))
        exit(0)
    return optimizer


def create_optimizer(configuration):
    """Create an optimizer given the configuration.
    This is the main interface between this package and train.py/validate.py
    """
    optimizer = find_optimizer_using_name(configuration['optimizer_name'])
    instance = optimizer(configuration)
    print("optimizer [{0}] was created".format(type(instance).__name__))
    return instance