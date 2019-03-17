import numpy as np
import matplotlib.pyplot as plt

# IKpy imports
from ikpy import chain
from ikpy import plot_utils


def test_ergo(resources_path, interactive):
    a = chain.Chain.from_urdf_file(resources_path + "/poppy_ergo.URDF")
    target = [0.1, -0.2, 0.1]
    frame_target = np.eye(4)
    frame_target[:3, 3] = target
    joints = [0] * len(a.links)
    ik = a.inverse_kinematics(frame_target, initial_position=joints)

    ax = plot_utils.init_3d_figure()
    a.plot(ik, ax, target=target)
    plt.savefig("out/ergo.png")

    if interactive:
        plot_utils.show_figure()
