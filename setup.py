"""Setup script for LSTM project."""

from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['setuptools>=40.2.0',
                     'matplotlib>=2.2.3',
                     'numpy>=1.15.1',
                     'tensorflow>=1.12.0']

setup(
    name='LSTM',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    description='LSTM network training project',
)
