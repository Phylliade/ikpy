# IKPy #
![Travis-CI](https://travis-ci.org/Phylliade/ikpy.svg?branch=master)
[![PyPI](https://img.shields.io/pypi/v/ikpy.svg)](https://pypi.python.org/pypi/ikpy/)
[![Documentation Status](https://readthedocs.org/projects/ikpy/badge/?version=latest)](http://ikpy.readthedocs.org/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/pyversions/ikpy/.svg)](https://pypi.python.org/pypi/ikpy/)
[![PyPI](https://img.shields.io/pypi/dm/ikpy.svg)](https://pypi.python.org/pypi/ikpy/)
[![Binder](http://mybinder.org/badge.svg)](http://mybinder.org/repo/Phylliade/ikpy)

![demo](two_arms.png)

# Features
With IKPy, you can :

* Compute the **Inverse Kinematics** of every existing robot.
* Define your kinematic chain using **arbitrary representations** : DH (Denavit–Hartenberg), URDF standard, custom...
* Automaticly import a kinematic chain from a **URDF file**.
* **Plot** your kinematic chain : no need to use a real robot (or a simulator) to test your algorithms!
* Define your own Inverse Kinematics methods

Moreover, IKPy is a **pure-Python library** : the install is a matter of seconds, and no compiling is required.

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

# Quickstart
Follow this IPython [notebook](https://github.com/Phylliade/ikpy/tree/master/notebooks/ikpy/Quickstart.ipynb).

# First concepts
Finished the quick start guide? Go [here](https://github.com/Phylliade/ikpy/tree/master/src/ikpy/README.md)!

# Guides and Tutorials
Go the the [src](https://github.com/Phylliade/ikpy/tree/master/src/ikpy) folder and read the .md files. It should introduce you to the basics concepts of IKPy.

You can find tutorials in the IPython [notebooks](https://github.com/Phylliade/ikpy/tree/master/notebooks/ikpy).

# API Documentation
An extensive documentation of the API can be found [here](http://ikpy.readthedocs.org).


# Dependencies and compatibility
The library can work with both versions of Python (2.7 and 3.x).
It requires numpy and scipy.

Sympy is not mandatory, but highly recommended, for fast hybrid computations, that's why it is installed by default.

Matplotlib is optional : it is used to plot (in 3D) your models.


# Contributing
IKPy is designed to be easily customisable : you can add your own IK methods or FK representations.

Contributations are welcome : if you have a hyperpower IK method, don't hesitate to propose for adding in the lib!
