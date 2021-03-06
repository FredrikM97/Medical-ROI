import pytest
from src.neural_network import architectures
from src.neural_network.architectures.resnet import ResNet
def test_create_architecture():
    #architectures.create_architecture
    pass

def test_find_architecture():
    with pytest.raises(ZeroDivisionError):
        assert architectures.find_architecture_using_name('not exist')

def test_vgg():
    pass

#@pytest.mark.parametrize("test_input,expected", [("resnet50", 8), (ResNet)])
def test_resnet50():
    #architectures.
    pass

def test_testModel():
    pass