import pytest
from src.utils import load

@pytest.fixture
def datadir():
    return 'data/SPM_categorised/AIH/'

def test_find_using_name():
    pass

def test_load_configs():
    # Test with missing path 
    assert load.load_configs('') == {}
    
    # Test with existing path
    assert len(load.load_configs('conf')) >0
    
def test_load_xml():
    pass
        
def test_load_nifti():
    pass

def test_load_files(datadir):
    assert len(load.load_files(datadir + '/')) > 0

    assert len(load.load_files(datadir)) > 0