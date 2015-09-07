import matplotlib.pyplot
import forward_kinematics


def plot_robot(robot_parameters, nodes_angles, ax):
    """Dessine le robot"""

    matplotlib.pyplot.axis('equal')
    (points, axes) = forward_kinematics.get_nodes(robot_parameters, nodes_angles)
    # print(points)

    ax.plot([-1, 0], [0, 0], [0, 0])
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
