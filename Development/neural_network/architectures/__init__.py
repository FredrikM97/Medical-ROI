"""This package contains modules related to objective functions, optimizations, and network architectures.
To add a custom model class called 'dummy', you need to add a file called 'dummy_model.py' and define a subclass DummyModel inherited from BaseModel.
You need to implement the following five functions:
    -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, configuration).
    -- <set_input>:                     unpack data from dataset and apply preprocessing.
    -- <forward>:                       produce intermediate results.
    -- <optimize_parameters>:           calculate loss, gradients, and update network weights.
In the function <__init__>, you need to define four lists:
    -- self.network_names (str list):       define networks used in our training.
    -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them.
"""

import importlib
import torch.nn as nn 
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
    model = find_architecture_using_name(configuration['architecture'])
    instance = model(**configuration)
    print("architecture [{0}] was created".format(type(instance).__name__))
    return instance