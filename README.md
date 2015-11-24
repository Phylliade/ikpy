# Poppy Inverse Kinematics #
![Travis-CI](https://travis-ci.org/Phylliade/poppy-inverse-kinematics.svg?branch=master)
[![PyPI](https://img.shields.io/pypi/v/poppy_inverse_kinematics.svg)](https://pypi.python.org/pypi/poppy_inverse_kinematics/)
[![PyPI](https://img.shields.io/pypi/pyversions/poppy_inverse_kinematics/.svg)](https://pypi.python.org/pypi/poppy_inverse_kinematics/)
[![PyPI](https://img.shields.io/pypi/dm/poppy_inverse_kinematics.svg)](https://pypi.python.org/pypi/poppy_inverse_kinematics/)
[![Binder](http://mybinder.org/badge.svg)](http://mybinder.org/repo/Phylliade/poppy-inverse-kinematics)
# Installation
You have two options :


1. From PyPI (recommended) : Simply run :
  ```
  pip install poppy-inverse-kinematics
  ```
2. From source : First download and extract the archive, the run :
  ```
  pip install ./
  ```    
  NB : You must have the proper rights to execute this command

# API Documentation
An exhaustive documentation of the API can be found [here](http://poppy-inverse-kinematics.readthedocs.org).


# Dependencies and compatibility
The library can work with both versions of Python (2.7 and 3.x).
It requires numpy and scipy.

Sympy is not mandatory, but highly recommended for fast hybrid computations, that's why it is installed by default.

Matplotlib is optional : it is used to plot your models.
