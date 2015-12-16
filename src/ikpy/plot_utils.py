# coding= utf8
import matplotlib.pyplot
import numpy as np
from . import forward_kinematics
from . import inverse_kinematics
from . import geometry_utils
import matplotlib.animation
from mpl_toolkits.mplot3d import Axes3D


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
        (rotation, node) = geometry_utils.from_transformation_matrix(transformation_matrixes[index])
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


def plot_robot(robot_parameters, nodes_angles, ax, representation, model_type):
    """Dessine le robot"""

    nodes = forward_kinematics.get_nodes(robot_parameters, nodes_angles, representation=representation, model_type=model_type)
    points = nodes["positions"]
    axes = nodes["rotation_axes"]
    # print(points)

    # Plot des axes entre les noeuds
    ax.plot([x[0] for x in points], [x[1]
            for x in points], [x[2] for x in points])
    # Plot des noeuds
    ax.scatter([x[0] for x in points], [x[1]
               for x in points], [x[2] for x in points])

    # Plot des axes de rotation
    for index, axe in enumerate(axes):
        ax.plot([points[index][0], axe[0] + points[index][0]], [points[index][1],
                axe[1] + points[index][1]], [points[index][2], axe[2] + points[index][2]])


def plot_target(target, ax):
    """Ajoute la target au plot"""
    ax.scatter(target[0], target[1], target[2], c="red", s=80)


def plot_target_trajectory(targets_x, targets_y, targets_z, ax):
    """Ajoute la trajectoire (liste des targets) au plot"""
    ax.scatter(targets_x, targets_y, targets_z)


def update_line(num, robot_parameters, nodes_angles_list, line, representation, model_type):
    nodes = forward_kinematics.get_nodes(robot_parameters, nodes_angles_list[num], representation=representation, model_type=model_type)
    points = nodes["positions"]
    line.set_data([x[0] for x in points], [x[1] for x in points])
    line.set_3d_properties([x[2] for x in points])


def animate_trajectory(robot_parameters, representation, model_type, starting_nodes_angles, targets_x, targets_y, targets_z, figure, bounds=None):
    ax = figure.add_subplot(111, projection='3d')

    # Création d'un objet line
    line = ax.plot([0, 0], [0, 0], [0, 0])[0]

    # Plot de la trajectoire et du repère
    plot_target_trajectory(targets_x, targets_y, targets_z, ax)
    plot_basis(robot_parameters, ax)

    # Liste des angles qui satisfont
    IK_angles = inverse_kinematics.inverse_kinematic_trajectory(robot_parameters, starting_nodes_angles, targets_x, targets_y, targets_z, bounds)
    return matplotlib.animation.FuncAnimation(figure, update_line, len(IK_angles), fargs=(robot_parameters, IK_angles, line, representation, model_type), interval=50)


def init_3d_figure():
    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    return ax


def show_figure():
    matplotlib.pyplot.show()
