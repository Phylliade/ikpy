# Inverse Kinematics #

# Active links
In IKPy, you can activate or deactivate at will some

For example, in this 4 links Chain, by specifying the mask :
```
[True, True, False, True]
```
You will activate every link, but the third one :
![](https://github.com/Phylliade/ikpy/blob/master/tutorials/ikpy/link-mask.png)

To use the link, mask, use the parameter `active_links_mask` when creating the [Chain](http://ikpy.readthedocs.org/en/latest/chain.html) object.

# Initial position
To compute the Inverse Kinematics, the algorithm will need the initial position of the chain. You can pass it by using the `initial_position` parameter of the `inverse_kinematics` method.
The expected datatype is exactly the same as the expected value of `joints` of the `forward_kinematics` method.


This is a very important parameter, and can have huge consequences on the computations (in terms of duration and the returned solution)
If you don't provide it, IKPy will take an array filled with zeros as the initial position.
