import matplotlib.pyplot
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def Rx_matrix(theta):
    """Matrice de rotation autour de l'axe X"""
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])


def Rz_matrix(theta):
    """Matrice de rotation autour de l'axe Z"""
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])


def rotation_matrix(phi, theta, psi):
    """Retourne la matrice de rotation décrite par les angles d'Euler donnés en paramètres"""
    return np.dot(Rz_matrix(phi), np.dot(Rx_matrix(theta), Rz_matrix(psi)))


def get_nodes(robot_parameters, nodes_angles):
    """Renvoie la liste des position des noeuds du robot, à partir de ses paramètres, et de la liste des angles
    La liste a len(robot_parameters) + 1 éléments et commence par (0,0,0)"""
    full_list = [
        (x, y, z, t) for ((x, y, z), t) in zip(robot_parameters, nodes_angles)
    ]

    #  Initialisations
    # Liste des positions des noeuds
    pos_list = []
    pos = np.array([0, 0, 0])
    pos_list.append(pos)
    # Matrice de rotation
    frame_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # Liste des axes de rotation de chaque noeud
    rotation_axe = np.array([0, 0, 1])
    rotation_axes = []

    # Calcul des positions de chaque noeud
    for index, (phi, theta, translation_vector, psi) in enumerate(full_list):
        pos_index = index + 1
        origin = pos_list[pos_index - 1]

        # Calcul de la nouvelle matrice de rotation
        frame_matrix = np.dot(frame_matrix, rotation_matrix(phi, theta, psi))
        # print(index, frame_matrix)

        # Calcul de la position du noeud actuel
        pos_relat = np.array(translation_vector)
        pos_list.append(np.dot(frame_matrix, pos_relat) + origin)

        # Calcul des coordonnées de l'axe de rotation
        rotation_axe = np.dot(frame_matrix, np.array([0, 0, 1]))
        rotation_axes.append(rotation_axe)

    return pos_list, rotation_axes


def plot_robot(robot_parameters, nodes_angles, ax):
    """Dessine le robot"""

    matplotlib.pyplot.axis('equal')
    (points, axes) = get_nodes(robot_parameters, nodes_angles)
    # print(points)

    # Plot des axes entre les noeuds
    ax.plot([x[0] for x in points], [x[1] for x in points], [x[2] for x in points])
    # Plot des noeuds
    ax.scatter([x[0] for x in points], [x[1] for x in points], [x[2] for x in points])

    # Plot des axes de rotation
    for index, axe in enumerate(axes):
        ax.plot([points[index][0], axe[0] + points[index][0]], [points[index][1], axe[1] + points[index][1]], [points[index][2], axe[2] + points[index][2]])


if (__name__ == "__main__"):
    # Paramètres du robot
    robot_parameters = [(0, 0, [0, 0, 4]), (0, 0, [3, 0, 0]), (0, np.pi / 2, [1, 0, 0])]
    nodes_angles = [0, np.pi / 4, np.pi / 2]

    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot_robot(robot_parameters, nodes_angles, ax)
    matplotlib.pyplot.show()
