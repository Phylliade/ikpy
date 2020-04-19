import os
import matplotlib.pyplot as plt

# ikpy imports
from ikpy import chain
from ikpy.utils import plot
from ikpy.urdf import URDF
from ikpy.urdf.utils import plot_urdf_tree


def test_urdf_chain(resources_path, interactive):
    """Test that we can open chain from a URDF file"""
    chain1 = chain.Chain.from_urdf_file(
        os.path.join(resources_path, "poppy_torso/poppy_torso.URDF"),
        base_elements=[
            "base", "abs_z", "spine", "bust_y", "bust_motors", "bust_x",
            "chest", "r_shoulder_y"
        ],
        last_link_vector=[0, 0.18, 0],
        active_links_mask=[
            False, False, False, False, True, True, True, True, True
        ])

    joints = [0] * len(chain1.links)
    joints[-4] = 0
    fig, ax = plot.init_3d_figure()
    chain1.plot(joints, ax)
    plt.savefig("out/chain1.png")
    if interactive:
        plt.show()


def test_urdf_parser(resources_path):
    """Test the correctness of a URDF parser"""
    urdf_file = os.path.join(resources_path, "poppy_torso/poppy_torso.URDF")
    base_elements = [
            "base", "abs_z", "spine", "bust_y", "bust_motors", "bust_x",
            "chest", "r_shoulder_y"
        ]
    last_link_vector = [0, 0.18, 0]

    links = URDF.get_urdf_parameters(urdf_file, base_elements=base_elements, last_link_vector=last_link_vector)

    assert len(links) == len(base_elements)


def test_plot_urdf_tree(baxter_urdf):
    dot = plot_urdf_tree(baxter_urdf, "./out/baxter")
