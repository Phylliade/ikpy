# coding= utf8
import matplotlib.pyplot
import numpy as np
import matplotlib.animation

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
    ax.legend()


def plot_chain(chain, joints, ax, target=None, show=False, length=1):
    """Plot the chain"""
    # List of nodes
    nodes = []
    # And rotation axes, as pairs of points
    rotation_axes = []

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

    # Plot the chain
    ax.plot([x[0] for x in nodes], [x[1] for x in nodes], [x[2] for x in nodes], linewidth=5)
    # Plot of the nodes of the chain
    ax.scatter([x[0] for x in nodes], [x[1] for x in nodes], [x[2] for x in nodes], s=55)

    # Plot the rotation axes
    for (node, axe) in rotation_axes:
        # The last link doesn't need a rotation axe
        ax.plot([node[0], axe[0]], [node[1], axe[1]], [node[2], axe[2]])

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

    print(np.dot(axes[0] - node[:3], axes[1] - node[:3]))

    # Plot the rotation axes
    # NOTE: Could be interesting to plot arrows instead of lines,
    # as in https://stackoverflow.com/questions/22867620/putting-arrowheads-on-vectors-in-matplotlibs-3d-plot
    for index, axe in enumerate(axes):
        ax.plot([node[0], axe[0]], [node[1], axe[1]], [node[2], axe[2]], linestyle='dashed', c=directions_colors[index])


def plot_target(target, ax):
    """Add the target to the plot"""
    ax.scatter(target[0], target[1], target[2], c="red", s=80)


def plot_target_trajectory(targets_x, targets_y, targets_z, ax):
    """Add the trajectory (liste of targets) to the plot"""
    ax.scatter(targets_x, targets_y, targets_z)


def init_3d_figure():
    from mpl_toolkits.mplot3d import axes3d, Axes3D
    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    return fig, ax


def show_figure():
    matplotlib.pyplot.show()
