# Tutorials

This section provides guides and notebooks used on the wiki.

If you're here to follow the tutorials, not develop them, jump to the [wiki](https://github.com/Phylliade/ikpy/wiki).

## Notebooks

There are also some notebooks to help you getting started. Currently available notebooks :

* [Quickstart](./tutorials/Quickstart.ipynb)
* [Inverse Kinematics with the Ergo Jr](./tutorials/Inverse%20Kinematics%20with%20ErgoJr.ipynb) : A demo of IKPy using the ErgoJr
* [Moving the Poppy Torso using Inverse Kinematics](./tutorials/Moving%20the%20Poppy%20Torso%20using%20Inverse%20Kinematics.ipynb)

The two Poppy notebooks (Ergo Jr and Torso) drive a simulated robot through [pypot](https://github.com/poppy-project/pypot) and its `poppy.creatures` packages, which wrap IKPy chains and talk to the V-REP/CoppeliaSim simulator. They need those packages and a running simulator; IKPy itself only provides the kinematics (the Poppy URDF files live in `resources/` and are exercised by `tests/test_poppy_robots.py`).

