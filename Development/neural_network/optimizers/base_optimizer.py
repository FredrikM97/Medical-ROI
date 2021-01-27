import math
import torch
from torch import optim 
from abc import ABC, abstractmethod

class BaseOptimizer(optim.Optimizer, ABC):
    def __init__(self,configuration):
        self.configuration = configuration
    
    @abstractmethod
    def step(self, closure=None):
        raise NotImplemented