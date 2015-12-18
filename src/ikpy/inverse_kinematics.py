# coding= utf8
import scipy.optimize
import numpy as np


def inverse_kinematic_optimization(chain, target, starting_nodes_angles, bounds=None, first_active_joint=0, regularization_parameter=None, max_iter=None, **kwargs):
    """Computes the inverse kinematic on the specified target with an optimization method"""
    # print("Sarting optimisation with bounds : ", bounds)

    if starting_nodes_angles is None:
        raise ValueError("starting_nodes_angles must be specified")

    # Compute squared distance to target
    def optimize_target(x):
        y = np.append(starting_nodes_angles[:first_active_joint], x)
        squared_distance = np.linalg.norm(chain.forward_kinematics(y)[:3, -1] - target)
        return squared_distance

    # If a regularization is selected
    if regularization_parameter is not None:
        def optimize_total(x):
            regularization = np.linalg.norm(x - starting_nodes_angles[first_active_joint:])
            return optimize_target(x) + regularization_parameter * regularization
    else:
        def optimize_total(x):
            return optimize_target(x)

    # Compute bounds
    real_bounds = [link.bounds for link in chain.links]
    real_bounds = real_bounds[first_active_joint:]

    options = {}
    # Manage iterations maximum
    if max_iter is not None:
        options["maxiter"] = max_iter

    # Utilisation d'une optimisation L-BFGS-B
    res = scipy.optimize.minimize(optimize_total, starting_nodes_angles[first_active_joint:], method='L-BFGS-B', bounds=real_bounds, options=options)

    print("Inverse kinematic optimisation OK, done in {} iterations".format(res.nit))
    if first_active_joint == 0:
        return(res.x)
    else:
        return(np.append(starting_nodes_angles[:first_active_joint], res.x))
