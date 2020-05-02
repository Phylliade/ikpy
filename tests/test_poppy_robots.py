import numpy as np
import matplotlib.pyplot as plt

# IKpy imports
from ikpy import chain
from ikpy.utils import plot


def test_ergo(resources_path, interactive):
    a = chain.Chain.from_urdf_file(resources_path + "/poppy_ergo.URDF")
    target = [0.1, -0.2, 0.1]
    frame_target = np.eye(4)
    frame_target[:3, 3] = target
    joints = [0] * len(a.links)
    ik = a.inverse_kinematics_frame(frame_target, initial_position=joints)

    fig, ax = plot.init_3d_figure()
    a.plot(ik, ax, target=target)
    plt.savefig("out/ergo.png")

    if interactive:
        plot.show_figure()
