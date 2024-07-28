from ikpy.chain import Chain
from ikpy.link import OriginLink, DHLink
from ikpy.utils import plot

import matplotlib.pyplot as plt

import numpy as np
from math import pi

class UR10():
    def __init__(self):
        self.robot_name = 'UR10'
        self.home_config = [0, -pi/2, 0, -pi/2, 0, 0] 
        self.dh_params = np.array([
            [  0.1273, 0., pi/2, 0.],
            [  0., -0.612, 0, 0.],
            [  0., -0.5723, 0, 0.],
            [  0.163941, 0.,  pi/2, 0.],
            [  0.1147, 0.,  -pi/2, 0.],
            [  0.0922, 0., 0, 0.]])   
        
        self.joint_limits =  [
                        (-360, 360),  
                        (-360, 360),  
                        (-360, 360),  
                        (-360, 360),  
                        (-360, 360),  
                        (-360, 360)] 
        
def create_dh_robot(robot):
    # Create a list of links for the robot
    links = []
    for i, dh in enumerate(robot.dh_params):
        link = DHLink(d=dh[0], a=dh[1], alpha=dh[2], theta=dh[3], length=abs(dh[1]))
        link.bounds = robot.joint_limits[i]
        links.append(link)

    # Create a chain using the robot links
    chain = Chain(links, name=robot.robot_name)
    return chain


def test_dh_chain():
    robot = UR10()
    chain = create_dh_robot(robot)
    frame = [[1.112918581, -0.209413742, 0.19382176], [0.0, 1.0, 0.0]]
    target_position, target_orientation = frame
    joint_angles = chain.inverse_kinematics(target_position=target_position, target_orientation=target_orientation, orientation_mode="Z")

    print(joint_angles)

    fig, ax = plot.init_3d_figure()
    chain.plot(robot.home_config, ax)
    chain.plot(joint_angles, ax)
    plt.savefig("out/UR10.png")
