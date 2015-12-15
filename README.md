# IKPy #
![Travis-CI](https://travis-ci.org/Phylliade/ikpy.svg?branch=master)
[![PyPI](https://img.shields.io/pypi/v/ikpy.svg)](https://pypi.python.org/pypi/ikpy/)
[![Documentation Status](https://readthedocs.org/projects/ikpy/badge/?version=latest)](http://ikpy.readthedocs.org/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/pyversions/ikpy/.svg)](https://pypi.python.org/pypi/ikpy/)
[![PyPI](https://img.shields.io/pypi/dm/ikpy.svg)](https://pypi.python.org/pypi/ikpy/)
[![Binder](http://mybinder.org/badge.svg)](http://mybinder.org/repo/Phylliade/ikpy)

![demo](two_arms.png)

# Installation
You have two options :


1. From PyPI (recommended) : Simply run :

   ```bash
   pip install ikpy
   ```
2. From source : First download and extract the archive, then run :

   ```bash
   pip install ./
   ```    
   NB : You must have the proper rights to execute this command

# Tutorials
You can find tutorials in the IPython [notebooks](https://github.com/Phylliade/ikpy/tree/master/notebooks).

# API Documentation
An extensive documentation of the API can be found [here](http://ikpy.readthedocs.org).


# Dependencies and compatibility
The library can work with both versions of Python (2.7 and 3.x).
It requires numpy and scipy.

Sympy is not mandatory, but highly recommended, for fast hybrid computations, that's why it is installed by default.

Matplotlib is optional : it is used to plot (in 3D) your models.
