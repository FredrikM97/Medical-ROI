import os, sys, site
from pathlib import Path
from torch.utils import cpp_extension
from setuptools import setup, find_packages

setup(
    name="Master-thesis", 
    version='0.1',
    url='None',
    author='Fredrik test',
    author_email='None',
    description='Prioritize Informative Structures in 3D Brain Images',
    packages=find_packages(),
    include_package_data=True,
    #namespace_packages=['roi_al_extension_3d','roi_al_extension','nms_extension']
    #exclude=['tests','logs','data']
)