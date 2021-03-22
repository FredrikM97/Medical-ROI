import pytest
from src.classifier import models
from src.classifier.models.resnet import ResNet

def test_create_model():
    #architectures.create_architecture
    pass

def test_find_model():
    with pytest.raises(AttributeError):
        assert models.find_model_using_name('not exist')

def test_vgg():
    pass

#@pytest.mark.parametrize("test_input,expected", [("resnet50", 8), (ResNet)])
def test_resnet50():
    #architectures.
    pass

def test_testModel():
    pass