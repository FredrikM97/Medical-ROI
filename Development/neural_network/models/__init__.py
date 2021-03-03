import importlib
from pytorch_lightning import LightningModule

BASEDIR = 'neural_network.'
def find_model_using_name(model_name):
    """Import the module "models/[model_name]_model.py".
        In the file, the class called DatasetNameModel() will
        be instantiated. It has to be a subclass of BaseModel,
        and it is case-insensitive.
    """
    model_filename = BASEDIR+"models." + model_name + "_model"
    modellib = importlib.import_module(model_filename)
    model = None
    target_model_name = model_name.replace('_', '') + 'model'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() \
           and issubclass(cls, LightningModule):
            model = cls

    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (model_filename, target_model_name))
        exit(0)

    return model

def create_model(**configuration:dict):
    """Create a model given the configuration.
    This is the main interface between this package and train.py/validate.py
    """
    model = find_model_using_name(configuration['model_name'])
    instance = model(**configuration)
    print("model [{0}] was created".format(type(instance).__name__))
    return instance