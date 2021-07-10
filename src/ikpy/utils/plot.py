# coding= utf8
import numpy as np
import matplotlib.animation
import matplotlib.pyplot as plt  # noqa: F401

# Ikpy imports
from ikpy.utils import geometry


# Colors of each directions axes. For ex X is green
directions_colors = ["green", "cyan", "orange"]


def plot_basis(ax, arm_length=1):
    """Plot a frame fitted to the robot size"""

    ax.set_xlim3d([-1.0, 1.0])
    ax.set_xlabel('X')

    ax.set_ylim3d([-1.0, 1.0])
    ax.set_ylabel('Y')

    ax.set_zlim3d([-1.0, 1.0])
    ax.set_zlabel('Z')

    # Plot du repère
    # Sa taille est relative à la taille du bras
    ax.plot([0, arm_length * 1.5], [0, 0], [0, 0], c=directions_colors[0], label="X")
    ax.plot([0, 0], [0, arm_length * 1.5], [0, 0], c=directions_colors[1], label="Y")
    ax.plot([0, 0], [0, 0], [0, arm_length * 1.5], c=directions_colors[2], label="Z")


def plot_chain(chain, joints, ax, name="chain"):
    """Plot the chain"""
    # List of nodes
    nodes = []
    # And rotation/translation axes, as pairs of points
    rotation_axes = []
    translation_axes = []

    transformation_matrixes = chain.forward_kinematics(joints, full_kinematics=True)

    # Get the nodes and the orientation from the tranformation matrix
    for (index, link) in enumerate(chain.links):
        (node, orientation) = geometry.from_transformation_matrix(transformation_matrixes[index])

        # Add node corresponding to the link
        nodes.append(node)

        # Add rotation axis if present
        if link.has_rotation:
            rotation_axis = link.get_rotation_axis()
            if index == 0:
                rotation_axes.append((node, rotation_axis))
            else:
                rotation_axes.append((node, geometry.homogeneous_to_cartesian_vectors(np.dot(transformation_matrixes[index - 1], rotation_axis))))

        # Add translation axis if present
        if link.has_translation:
            translation_axis = link.get_translation_axis()
            if index == 0:
                translation_axes.append((node, translation_axis))
            else:
                translation_axes.append((node, geometry.homogeneous_to_cartesian_vectors(np.dot(transformation_matrixes[index - 1], translation_axis))))

    # Plot the chain
    lines = ax.plot([x[0] for x in nodes], [x[1] for x in nodes], [x[2] for x in nodes], linewidth=5, label=name)
    # Plot of the nodes of the chain
    # Note: Be sure that the nodes have the same color as the chain
    ax.scatter([x[0] for x in nodes], [x[1] for x in nodes], [x[2] for x in nodes], s=55, c=lines[0].get_color())

    # Plot the rotation axes
    for (node, axe) in rotation_axes:
        # The last link doesn't need a rotation axe
        # Note: Be sure that the rotation axes have the same color as the chain
        ax.plot([node[0], axe[0]], [node[1], axe[1]], [node[2], axe[2]], c=lines[0].get_color())

    # Plot the translation axes
    for (node, axe) in translation_axes:
        # The last link doesn't need a rotation axe
        # Note: Be sure that the rotation axes have the same color as the chain
        ax.plot([node[0], axe[0]], [node[1], axe[1]], [node[2], axe[2]], c=lines[0].get_color(), linestyle='dotted', linewidth=2.5)

    # Plot the frame of the last joint
    plot_frame(transformation_matrixes[-1], ax, length=chain.links[-1].length)


def plot_frame(frame_matrix, ax, length=1):
    # Plot the last joint as a solid
    (node, rotation) = geometry.from_transformation_matrix(frame_matrix)
    axes = [
        geometry.homogeneous_to_cartesian_vectors(np.dot(frame_matrix, [length, 0, 0, 1])),
        geometry.homogeneous_to_cartesian_vectors(np.dot(frame_matrix, [0, length, 0, 1])),
        geometry.homogeneous_to_cartesian_vectors(np.dot(frame_matrix, [0, 0, length, 1]))
    ]

    for index, axe in enumerate(axes):
        ax.plot([node[0], axe[0]], [node[1], axe[1]], [node[2], axe[2]], linestyle='dashed', c=directions_colors[index])


def plot_target(target, ax):
    """Add the target to the plot"""
    ax.scatter(target[0], target[1], target[2], c="red", s=80)


def plot_target_trajectory(targets_x, targets_y, targets_z, ax):
    """Add the trajectory (liste of targets) to the plot"""
    ax.scatter(targets_x, targets_y, targets_z)


def init_3d_figure():
    from mpl_toolkits.mplot3d import axes3d, Axes3D  # noqa: F401
    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Add the initial frame
    plot_basis(ax)
    return fig, ax


def show_figure():
    matplotlib.pyplot.show()
