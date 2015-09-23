# coding: utf8
import numpy as np


def euler_from_unit_vector(x, y, z):
    if x != 0:
        theta = np.arctan(y / x)
    else:
        theta = np.pi / 2
    phi = np.arccos(z)
    return(theta, np.pi / 2 - theta, phi)


def euler_from_URDF_parameters(vector_parameters):
    return [(euler_from_unit_vector(x), y) for (x, y) in vector_parameters]


def get_robot_length(robot_parameters):
    """Calcul la longueur du robot (tendu)"""
    translations_vectors = [x[2] for x in robot_parameters]
    joints_lengths = [np.sqrt(sum([x**2 for x in vector])) for vector in translations_vectors]
    return sum(joints_lengths)


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


def FK_jacobian(robot_parameters, nodes_angles):
    """Retourne le jacobien de la FK"""
    pass


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
    # Initiailisation de la matrice de changement de base
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

        arm_length = np.sqrt(sum([x**2 for x in translation_vector]))

        # Calcul des coordonnées de l'axe de rotation
        rotation_axe = np.dot(frame_matrix, np.array([0, 0, 1]) * arm_length)
        rotation_axes.append(rotation_axe)

    return pos_list, rotation_axes
