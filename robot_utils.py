import forward_kinematics


def robot_from_urdf_parameters(urdf_params):
    return [(forward_kinematics.euler_from_unit_vector(rot), trans) for (rot, trans) in urdf_params]
