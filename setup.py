#!/usr/bin/env python
from setuptools import setup

setup(name='poppy-inverse-kinematics',
      version='1.3.1',
      author="Pierre Manceron",
      description="An inverse kinematics library aiming performance and modularity",
      url="https://github.com/Phylliade/poppy-inverse-kinematics",
      license="GNU GENERAL PUBLIC LICENSE Version 3",
      packages=['poppy_inverse_kinematics', 'poppy_inverse_kinematics.resources'],
      package_dir={'': 'src'},
      package_data={'poppy_inverse_kinematics.resources': ['*']},
      setup_requires=['numpy'],
      install_requires=['numpy', 'scipy', 'sympy'],

      )
