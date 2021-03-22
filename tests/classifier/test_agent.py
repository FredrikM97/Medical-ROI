import pytest
from src.classifier.agent import Agent
from src.utils import load

@pytest.fixture
def config_name():
    assert len(load.load_configs('base')) > 0
    return 'base'

@pytest.fixture
def agent(config_name):
    return Agent(config_name)

def test_load_model(agent):
    agent.load_model()
    assert agent.model

def test_load_config(agent,config_name):
    agent.load_config(config_name)
    assert agent.config
    
def test_logger():
    assert agent.logger()

def test_callbacks():
    assert agent.callbacks()

def test_gpus():
    assert agent.gpus