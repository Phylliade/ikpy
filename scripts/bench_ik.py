import poppy_inverse_kinematics.robot_utils
import poppy_inverse_kinematics.model
import cProfile
import pstats
import timeit
import scripts_config

# Parameters
# Target
target = [1, 1, 1]
# Profile if True
profile = True
# Count elapsed time if True
bench = True
# Computation method
computation_method = "hybrid"


def setup():
    urdf_params = poppy_inverse_kinematics.robot_utils.get_urdf_parameters(scripts_config.urdf_file)
    robot = poppy_inverse_kinematics.model.Model(urdf_params, representation=scripts_config.representation, model_type=scripts_config.model_type, computation_method=computation_method)
    robot.target = target
    return robot


def bench_ik(robot):
    return timeit.timeit(lambda: robot.inverse_kinematic(), number=100)


def profile_ik(robot):
    cProfile.run("robot.inverse_kinematic()", "output/stats")
    p = pstats.Stats('output/stats')
    p.strip_dirs().sort_stats("cumulative").print_stats()


if __name__ == "__main__":
    # profile_robot()
    robot = setup()
    if profile:
        print(bench_ik(robot))
    if bench:
        profile_ik(robot)
