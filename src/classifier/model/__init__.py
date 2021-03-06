import importlib
from .resnet import *
from .vgg import *

BASEDIR = 'neural_network.'

def find_architecture_using_name(model_name):
    #The local module is already defined as 'architectures.'
    model_name = BASEDIR+'architectures.' + model_name
    components =  model_name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def create_architecture(**configuration:dict):
    """Create a model given the configuration.
    This is the main interface between this package and train.py/validate.py
    """
    
    model = find_architecture_using_name(configuration['architecture_name'])
    instance = model(**configuration)
        
    print("Architecture [{0}] was created".format(type(instance).__name__))
    return instance