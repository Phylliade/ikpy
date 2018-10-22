#!/usr/bin/env python
from setuptools import setup, find_packages
import os


def get_version():
    version = {}
    with open(os.path.join(os.path.dirname(__file__), "src/ikpy/_version.py")) as fp:
        exec(fp.read(), version)
    return version["__version__"]


setup(
    name='ikpy',
    version=get_version(),
    author="Pierre Manceron",
    description="An inverse kinematics library aiming performance and modularity",
    url="https://github.com/Phylliade/ikpy",
    license="GNU GENERAL PUBLIC LICENSE Version 3",
    packages=find_packages("src", exclude=("tests", "docs")),
    package_dir={'': 'src'},
    setup_requires=['numpy'],
    install_requires=['numpy', 'scipy', 'sympy'],
    extras_require={
        'plot': ["matplotlib"],
    },
    classifiers=[
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
    ])
