# URDF parsing and importing #

URDF parsing is done via the class method `from_urdf_file`.

There are multiple things to consider to parse a URDF file :

# URDF and IKPy conventions
An URDF robot is made of links and joints : each joint represents a motor, and is bound to two links : his parent link and his child link.

Be careful, the `links` of IKPy (and of the matlab toolbox) are more related to the URDF `joints`!

# Giving the base elements
Given the fact that a chain is linear, and If your robot is not a linear chain, for example an humanoid, then you must specify the elements that constitute your chain inthe URDF file



# Using a last_link_vector

# Specifying a chain mask
