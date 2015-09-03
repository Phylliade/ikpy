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
    """Renvoie la liste des position des noeuds du robot, à partir de ses paramètres, et de la liste des angles"""
    full_list = [
        (x, y, z, t) for ((x, y, z), t) in zip(np.asarray(robot_parameters), np.asarray(nodes_angles))
    ]

    pos_list = []
    pos = np.array([0, 0, 0])
    frame_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    pos_list.append(pos)
    rotation_axe = np.array([0, 0, 1])
    rotation_axes = []

    for index, (phi, theta, r, psi) in enumerate(full_list):
        pos_index = index + 1
        current_arm_length = r
        origin = pos_list[pos_index - 1]

        frame_matrix = np.dot(frame_matrix, rotation_matrix(phi, theta, psi))
        print(index, frame_matrix)
        pos_relat = np.array([1, 0, 0]) * current_arm_length
        pos_list.append(np.dot(frame_matrix, pos_relat) + origin)
        rotation_axe = np.dot(frame_matrix, np.array([0, 0, 1]))
        rotation_axes.append(rotation_axe)

    return pos_list, rotation_axes


robot_parameters = [(0, 0, 4), (0, 0, 3), (0, np.pi / 2, 1)]
nodes_angles = [0, np.pi / 2, np.pi / 2]
fig = matplotlib.pyplot.figure()
ax = fig.add_subplot(111, projection='3d')
matplotlib.pyplot.axis('equal')

(points, axes) = get_nodes(robot_parameters, nodes_angles)
print(points)

ax.plot([x[0] for x in points], [x[1] for x in points], [x[2] for x in points])
ax.scatter([x[0] for x in points], [x[1] for x in points], [x[2] for x in points])

for index, axe in enumerate(axes):
    ax.plot([points[index][0], axe[0] + points[index][0]], [points[index][1], axe[1] + points[index][1]], [points[index][2], axe[2] + points[index][2]])
matplotlib.pyplot.show()
