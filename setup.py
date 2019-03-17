#!/usr/bin/env python
from setuptools import setup, find_packages
import os
from os import path


def get_version():
    version = {}
    with open(os.path.join(os.path.dirname(__file__), "src/ikpy/_version.py")) as fp:
        exec(fp.read(), version)
    return version["__version__"]


# read the contents of your README file
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md')) as f:
    long_description = f.read()


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

    ],
    # Used in the Pypi page
    long_description=long_description,
    long_description_content_type='text/markdown'
)
