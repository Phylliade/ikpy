# coding= utf8
import matplotlib.pyplot
import numpy as np
import matplotlib.animation

# Ikpy imports
from . import geometry_utils


def plot_basis(ax, arm_length=1):
    """Plot le repère adapté à la taille du robot"""

    ax.set_xlim3d([-1.0, 1.0])
    ax.set_xlabel('X')

    ax.set_ylim3d([-1.0, 1.0])
    ax.set_ylabel('Y')

    ax.set_zlim3d([-1.0, 1.0])
    ax.set_zlabel('Z')

    # Plot du repère
    # Sa taille est relative à la taille du bras
    ax.plot([0, arm_length * 1.5], [0, 0], [0, 0])
    ax.plot([0, 0], [0, arm_length * 1.5], [0, 0])
    ax.plot([0, 0], [0, 0], [0, arm_length * 1.5])


def plot_chain(chain, joints, ax, target=None, show=False):
    """Plots the chain"""
    # LIst of nodes and orientations
    nodes = []
    axes = []

    transformation_matrixes = chain.forward_kinematics(joints, full_kinematics=True)

    # Get the nodes and the orientation from the tranformation matrix
    for (index, link) in enumerate(chain.links):
        (node, rotation) = geometry_utils.from_transformation_matrix(transformation_matrixes[index])
        nodes.append(node)
        rotation_axis = link._get_rotation_axis()
        if index == 0:
            axes.append(rotation_axis)
        else:
            axes.append(geometry_utils.homogeneous_to_cartesian_vectors(np.dot(transformation_matrixes[index - 1], rotation_axis)))

    # Plot the chain
    ax.plot([x[0] for x in nodes], [x[1] for x in nodes], [x[2] for x in nodes])
    # Plot of the nodes of the chain
    ax.scatter([x[0] for x in nodes], [x[1] for x in nodes], [x[2] for x in nodes])

    # Plot  rotation axes
    for index, axe in enumerate(axes):
        ax.plot([nodes[index][0], axe[0]], [nodes[index][1], axe[1]], [nodes[index][2], axe[2]])


def plot_target(target, ax):
    """Ajoute la target au plot"""
    ax.scatter(target[0], target[1], target[2], c="red", s=80)


def plot_target_trajectory(targets_x, targets_y, targets_z, ax):
    """Ajoute la trajectoire (liste des targets) au plot"""
    ax.scatter(targets_x, targets_y, targets_z)


def init_3d_figure():
    from mpl_toolkits.mplot3d import axes3d, Axes3D # noqa
    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    return ax


def show_figure():
    matplotlib.pyplot.show()
