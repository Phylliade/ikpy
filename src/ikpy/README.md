# Basic concepts #

This tutorial will introduce you to the basic concepts of IKPy.
You can test a live version in the corresponding [IPython Notebook](https://github.com/Phylliade/ikpy/blob/master/notebooks/Quickstart.ipynb)


# Chain

The chain is the main object you will work with : It is a list of links you can move, inspect, and plot.

## Creating a chain from a URDF file
One great feature of IKPy is that you can create your chains from a URDF file. If your file is URDF-compliant, it will work!
```
my_chain = Chain.from_urdf_file("poppy_ergo.URDF")
```

## Getting the position of your chain (aka Forward Kinematics, aka FK)
```
my_chain.forward_kinematics(joints_vector)
```

## Setting the position of your chain (aka Inverse Kinematics, aka IK)
```
my_chain.inverse_kinematics([1, 1, 1])
```
