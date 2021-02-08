import sys
sys.path.append('..')

import pytest
from utils import Adni


def test_adni_init_no_arg():
    adni = Adni()

def test_adni_init_root_arg():
    adni = Adni(root='../data')
    
def test_load_images():
    pass