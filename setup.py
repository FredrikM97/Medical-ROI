from setuptools import setup, find_packages,find_namespace_packages

setup(
    name="Prioritization of Informative Regions in PET Scansfor Classification of Alzheimer’s Disease", 
    version='1.0',
    url='None',
    author='Fredrik',
    author_email='fredrik9779@gmail.com',
    description='Master thesis for Prioritization of Informative Regions in PET Scansfor Classification of Alzheimer’s Disease in collaboration with Halmstad University.',
    packages=find_packages(),
    include_package_data=True,
    namespace_packages=find_namespace_packages(include=['dependencies'])
)