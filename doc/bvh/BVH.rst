:mod:`ikpy.bvh.BVH` module
----------------------------------------

BVH (Biovision Hierarchy) is the most common file format for motion capture data.
A file describes a skeleton (a tree of joints with offsets and rotation channels) and a motion
(the value of every channel, frame by frame).

IKPy builds a kinematic chain from any path of the skeleton, and replays the motion of the file on it.

For more information about BVH, see `Motion capture file formats explained <https://research.cs.wisc.edu/graphics/Courses/cs-838-1999/Jeff/BVH.html>`_.

.. automodule:: ikpy.bvh.BVH
    :members:
    :undoc-members:
    :show-inheritance:
