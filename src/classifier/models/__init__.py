import importlib
import types
from .resnet import *
from .vgg import *

BASEDIR = 'src.classifier.'

def find_model_using_name(model_name):
    from inspect import isclass
    from pkgutil import iter_modules
    from pathlib import Path
    from importlib import import_module

    # iterate through the modules in the current package
    package_dir = Path(__file__).resolve().parent
    for (_, module_name, _) in iter_modules([package_dir]):
        # import the module and iterate through its attributes
        module = import_module(f"{__name__}.{module_name}")
        for attribute_name in dir(module):
            attribute = getattr(module, attribute_name)
            if attribute_name == model_name and (isclass(attribute) or isinstance(attribute, types.FunctionType)):
                return attribute
            
def create_model(name:str=None, args:dict={}):
    """Create a model given the configuration.
    This is the main interface between this package and train.py/validate.py
    """
    
    model = find_model_using_name(name)
    instance = model(args)
    print("Architecture [{0}] was created".format(type(instance).__name__))
    return instance