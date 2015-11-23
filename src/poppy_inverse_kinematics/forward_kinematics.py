# coding= utf8
import numpy as np
import sympy


def euler_from_unit_vector(x, y, z):
    """Retourne les angles d'euler associés à un vecteur unitaire"""
    if x != 0:
        theta = np.arctan(y / x)
    else:
        theta = np.pi / 2
    phi = np.arccos(z)
    return(np.pi / 2 - theta, phi)


def euler_from_URDF_parameters(URDF_parameters):
    """Converts URDF_parameters to euler_parameters"""
    euler_parameters = []
    absolute_rotations = [param[0] for param in URDF_parameters]
    relative_euler_rotations = get_relative_angles(absolute_rotations)
    for i, (x, y) in enumerate(URDF_parameters):
        euler_parameters.append(
            ((relative_euler_rotations[i][0], relative_euler_rotations[i][1]), y))
    return euler_parameters


def get_relative_angles(absolute_vectors):
    """Convert a list of absolute_vectors to a list of vectors relative to the frame oriented by the previous in the list"""
    angles_list = []
    frame_matrix = np.eye(3)
    for absolute_vector in absolute_vectors:
        relative_vector = np.dot(np.transpose(frame_matrix), absolute_vector)
        (phi, theta) = euler_from_unit_vector(*relative_vector)
        angles_list.append((phi, theta))
        frame_matrix = np.dot(frame_matrix, rotation_matrix(phi, theta, 0))
    return angles_list


def get_robot_length(robot_parameters):
    """Calcul la longueur du robot (tendu)"""
    translations_vectors = [x[0] for x in robot_parameters]
    joints_lengths = [np.sqrt(sum([x**2 for x in vector]))
                      for vector in translations_vectors]
    return sum(joints_lengths)


def cartesian_to_homogeneous(cartesian_matrix, matrix_type="numpy"):
    """Converts a cartesian matrix to an homogenous matrix"""
    dimension_x, dimension_y = cartesian_matrix.shape
    # Square matrix
    # Manage different types fo input matrixes
    if matrix_type == "numpy":
        homogeneous_matrix = np.eye(dimension_x + 1)
    elif matrix_type == "sympy":
        homogeneous_matrix = sympy.eye(dimension_x + 1)
    # Add a column filled with 0 and finishing with 1 to the cartesian matrix to transform it into an homogeneous one
    homogeneous_matrix[:-1, :-1] = cartesian_matrix

    return homogeneous_matrix


def cartesian_to_homogeneous_vectors(cartesian_vector, matrix_type="numpy"):
    """Converts a cartesian vector to an homogenous vector"""
    dimension_x = cartesian_vector.shape[0]
    # Vector
    if matrix_type == "numpy":
        homogeneous_vector = np.zeros(dimension_x + 1)
        # Last item is a 1
        homogeneous_vector[-1] = 1
        homogeneous_vector[:-1] = cartesian_vector
    return homogeneous_vector


def homogeneous_to_cartesian_vectors(homogeneous_vector):
        """Converts a cartesian vector to an homogenous vector"""
        return homogeneous_vector[:-1]


def homogeneous_to_cartesian(homogeneous_matrix):
    """Converts a cartesian vector to an homogenous matrix"""
    # Remove the last column
    return homogeneous_matrix[:-1, :-1]


def homogeneous_transformation(homogeneous_matrix, cartesian_vector):
    """Returns the cartesian coordinates of the transformation_matrix applied to the input vector"""
    cartesian_pos = homogeneous_to_cartesian_vectors(np.dot(homogeneous_matrix, cartesian_to_homogeneous_vectors(cartesian_vector)))
    return cartesian_pos


def Rx_matrix(theta):
    """Rotation matrix around the X axis"""
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])


def Rz_matrix(theta):
    """Rotation matrix around the Z axis"""
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])


def symbolic_Rz_matrix(symbolic_theta):
    """Matrice symbolique de rotation autour de l'axe Z"""
    return sympy.Matrix([
        [sympy.cos(symbolic_theta), -sympy.sin(symbolic_theta), 0],
        [sympy.sin(symbolic_theta), sympy.cos(symbolic_theta), 0],
        [0, 0, 1]
    ])


def Ry_matrix(theta):
    """Rotation matrix around the Y axis"""
    return np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])


def rotation_matrix(phi, theta, psi):
    """Retourne la matrice de rotation décrite par les angles d'Euler donnés en paramètres"""
    return np.dot(Rz_matrix(phi), np.dot(Rx_matrix(theta), Rz_matrix(psi)))


def symbolic_rotation_matrix(phi, theta, symbolic_psi):
    """Retourne une matrice de rotation où psi est symbolique"""
    return sympy.Matrix(Rz_matrix(phi)) * sympy.Matrix(Rx_matrix(theta)) * symbolic_Rz_matrix(symbolic_psi)


def homogeneous_translation_matrix(trans_x, trans_y, trans_z):
    """Returns a translation matrix the homogeneous space"""
    return np.array([[1, 0, 0, trans_x], [0, 1, 0, trans_y], [0, 0, 1, trans_z], [0, 0, 0, 1]])


def axis_rotation_matrix(axis, theta):
    """Returns a translation matrix around the given axis"""
    [x, y, z] = axis
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [x**2 + (1 - x**2) * c, x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
        [x * y * (1 - c) + z * s, y ** 2 + (1 - y**2) * c, y * z * (1 - c) - x * s],
        [x * z * (1 - c) - y * s, y * z * (1 - c) + x * s, z**2 + (1 - z**2) * c]
    ])


def symbolic_axis_rotation_matrix(axis, symbolic_theta):
    """Returns a translation matrix around the given axis"""
    [x, y, z] = axis
    c = sympy.cos(symbolic_theta)
    s = sympy.sin(symbolic_theta)
    return sympy.Matrix([
        [x**2 + (1 - x**2) * c, x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
        [x * y * (1 - c) + z * s, y ** 2 + (1 - y**2) * c, y * z * (1 - c) - x * s],
        [x * z * (1 - c) - y * s, y * z * (1 - c) + x * s, z**2 + (1 - z**2) * c]
    ])


def rpy_matrix(roll, pitch, yaw):
    """Returns a rotation matrix described by the extrinsinc roll, pitch, yaw coordinates"""
    return np.dot(Rz_matrix(yaw), np.dot(Ry_matrix(pitch), Rx_matrix(roll)))


def symbolic_rpy_matrix(roll, pitch, symbolic_yaw):
    """Returns a symbolic rotation matrix described by the extrinsinc roll, pitch and symbolic yaw coordinates"""
    return symbolic_Rz_matrix(symbolic_yaw) * sympy.Matrix(Ry_matrix(pitch)) * sympy.Matrix(Rx_matrix(roll))


def FK_jacobian(robot_parameters, nodes_angles):
    """Retourne le jacobien de la FK"""
    pass


def get_nodes(robot_parameters, nodes_angles, representation="euler", model_type="custom"):
    """Renvoie la liste des position des noeuds du robot, à partir de ses paramètres, et de la liste des angles
    La liste a len(robot_parameters) + 1 éléments et commence par (0,0,0)"""
    if model_type == "custom":
        full_list = [
            (trans, rot, t) for ((trans, rot), t) in zip(robot_parameters, nodes_angles)
        ]

    elif model_type == "URDF":
        full_list = [
            (params["translation"], params["orientation"], params["rotation"], t) for (params, t) in zip(robot_parameters, nodes_angles)
        ]

    #  Initialisations
    # Liste des positions des noeuds
    pos_list = []
    pos = np.array([0, 0, 0])
    pos_list.append(pos)
    # Initialisation de la matrice de changement de base
    frame_matrix = np.eye(3)

    # Liste des axes de rotation de chaque noeud
    rotation_axe = np.array([0, 0, 1])
    rotation_axes = []
    # Origin rotation axe (virtual)
    rotation_axes.append(np.array([0, 0, 0.1]))

    # Calcul des positions de chaque noeud
    # print(representation, full_list)
    for index, params in enumerate(full_list):

        if model_type == "custom":
            (translation_vector, rot, psi) = params
        elif model_type == "URDF":
            (translation_vector, orientation, rot, psi) = params

        pos_index = index + 1
        origin = pos_list[pos_index - 1]

        # Calcul de la position du noeud actuel
        pos_relat = np.array(translation_vector)

        pos_list.append(np.dot(frame_matrix, pos_relat) + origin)

        if model_type == "URDF":
            frame_matrix = np.dot(frame_matrix, axis_rotation_matrix(rot, psi))
            # pos_relat = np.dot(axis_rotation_matrix(rot, psi), pos_relat)

        joint_length = np.sqrt(sum([x**2 for x in translation_vector]))

        # Calcul des coordonnées de l'axe de rotation
        if model_type == "custom":
            # En custom, l'axe de rotation relatif est confondu avec [0, 0, 1]
            relative_rotation_axe = np.array([0, 0, 1])
            rotation_axe = np.dot(frame_matrix, relative_rotation_axe * joint_length / 2)
            if representation == "euler":
                # Calcul de la nouvelle matrice de rotation
                frame_matrix = np.dot(frame_matrix, rotation_matrix(rot[0], rot[1], psi))
                # print(index, frame_matrix)

        elif model_type == "URDF":
            # En URDF, l'axe de rotation relatif est donné par rot
            relative_rotation_axe = np.array(rot)
            rotation_axe = np.dot(frame_matrix, relative_rotation_axe * joint_length / 2)
            if representation == "rpy":
                frame_matrix = np.dot(frame_matrix, rpy_matrix(*orientation))

        rotation_axes.append(rotation_axe)

    return {"positions": pos_list, "rotation_axes": rotation_axes}


def compute_transformation_symbolic(robot_parameters, representation="euler", model_type="custom", simplify=False):
    """Retourne la matrice de la forward_kinematic"""

    # Initial value of the frame matrix in homogeneous coordinates
    frame_matrix = sympy.eye(4)
    joint_angles = []

    # Calcul itératif de la matrice de le FK
    for index, params in enumerate(robot_parameters):
        if model_type == "custom":
            (translation_vector, rot) = params
        elif model_type == "URDF":
            translation_vector = params["translation"]
            orientation = params["orientation"]
            rot = params["rotation"]

        # Angle symbolique qui paramètre la rotation du joint en cours
        psi = sympy.symbols("psi_" + str(index))
        joint_angles.append(psi)

        # Apply translation matrix
        frame_matrix = frame_matrix * sympy.Matrix(homogeneous_translation_matrix(*translation_vector))
        if model_type == "URDF":
            # Apply rotation matrix
            frame_matrix = frame_matrix * cartesian_to_homogeneous(symbolic_axis_rotation_matrix(rot, psi), matrix_type="sympy")

        # Apply orientation matrix
        if model_type == "custom":
            if representation == "euler":
                # Calcul de la nouvelle matrice de rotation
                frame_matrix = frame_matrix * cartesian_to_homogeneous(symbolic_rotation_matrix(rot[0], rot[1], psi), matrix_type="sympy")

        elif model_type == "URDF":
            if representation == "rpy":
                frame_matrix = frame_matrix * cartesian_to_homogeneous(rpy_matrix(*orientation))

    if simplify:
        # Simplify the matrix
        frame_matrix = sympy.simplify(frame_matrix)

    # On retourne une fonction lambda de la FK
    return sympy.lambdify(joint_angles, frame_matrix)


def get_end_effector_symbolic(symbolic_transformation_matrix, nodes_angles):
    """Renvoie la position du end effector en fonction de la configuration des joints"""
    # On applique la matrice transformation au vecteur [0, 0, 0]
    return homogeneous_transformation(np.asarray(symbolic_transformation_matrix(*nodes_angles)), np.array([0, 0, 0]))


def compute_transformation_hybrid(robot_parameters, representation="euler", model_type="custom", simplify=False):
    """Returns the list of transformation matrixes for each joint"""

    joint_matrixes = []

    # Calcul itératif de la matrice de le FK
    for index, params in enumerate(robot_parameters):
        frame_matrix = sympy.eye(4)
        if model_type == "custom":
            (translation_vector, rot) = params
        elif model_type == "URDF":
            translation_vector = params["translation"]
            orientation = params["orientation"]
            rot = params["rotation"]

        # Angle symbolique qui paramètre la rotation du joint en cours
        psi = sympy.symbols("psi_" + str(index))

        # Apply translation matrix
        frame_matrix = frame_matrix * sympy.Matrix(homogeneous_translation_matrix(*translation_vector))
        if model_type == "URDF":
            # Apply rotation matrix
            frame_matrix = frame_matrix * cartesian_to_homogeneous(symbolic_axis_rotation_matrix(rot, psi), matrix_type="sympy")

        # Apply orientation matrix
        if model_type == "custom":
            if representation == "euler":
                # Calcul de la nouvelle matrice de rotation
                frame_matrix = frame_matrix * cartesian_to_homogeneous(symbolic_rotation_matrix(rot[0], rot[1], psi), matrix_type="sympy")

        elif model_type == "URDF":
            if representation == "rpy":
                frame_matrix = frame_matrix * cartesian_to_homogeneous(rpy_matrix(*orientation))

        if simplify:
            # Simplify the matrix
            frame_matrix = sympy.simplify(frame_matrix)

        # Save the joint transformation_matrix
        joint_matrixes.append(sympy.lambdify(psi, frame_matrix, "numpy"))

    # On retourne une fonction lambda de la FK
    return joint_matrixes


def get_end_effector_hybrid(symbolic_transformation_matrixes, nodes_angles):
    """Renvoie la position du end effector en fonction de la configuration des joints"""
    frame_matrix = np.eye(4)

    for index, (joint_matrix, joint_angle) in enumerate(zip(symbolic_transformation_matrixes, nodes_angles)):
        # Compute iteratively the position
        # NB : Use asarray to avoid old sympy problems
        frame_matrix = np.dot(frame_matrix, np.asarray(joint_matrix(joint_angle)))
    # Return the matrix origin
    return homogeneous_to_cartesian_vectors(np.dot(frame_matrix, np.array([0, 0, 0, 1])))


def compute_transformation(robot_parameters, method="default", representation="euler", model_type="custom", simplify=False):
    """Computes a transformation and returns an object depending of the selected method"""
    if method == "symbolic":
        # Symbolic method
        return compute_transformation_symbolic(robot_parameters, representation=representation, model_type=model_type, simplify=simplify)
    elif method == "hybrid":
        # Hybrid method
        return compute_transformation_hybrid(robot_parameters, representation=representation, model_type=model_type, simplify=simplify)
    else:
        # Default
        return None


def get_end_effector(nodes_angles, method="default", transformation_lambda=None, **kwargs):
    """Returns the a position of the end-effoctor, computed with the selected method"""
    if method == "hybrid":
        return get_end_effector_hybrid(transformation_lambda, nodes_angles)
    elif method == "symbolic":
        return get_end_effector_symbolic(transformation_lambda, nodes_angles)
    else:
        # Default method
        return get_nodes(kwargs["robot_parameters"], nodes_angles, representation=kwargs["representation"], model_type=kwargs["model_type"])["positions"][-1]
