import numpy as np

classical_arm_parameters = [(0, 0, [4, 0, 0]), (0, np.pi / 2, [3, 0, 0]), (0, np.pi / 2, [1, 0, 0])]
classical_arm_default_angles = [0, 0, np.pi / 2]

third_arm_parameters = [(np.pi / 2, np.pi / 2, [0, 0, 0.025]), (np.pi / 2, np.pi / 2, [0, 0.045, 0]), (0, - np.pi / 2, [0, 0, 0.105]), (0, np.pi / 2, [0, 0.15, 0]), (0, -np.pi / 2, [0, 0, 0.035]), (0, np.pi / 2, [0, 0.05, 0]), (np.pi / 2, np.pi / 2, [0.045, 0, 0]), (np.pi / 2, np.pi / 2, [0., 0, 0.03]), (0, -np.pi / 2, [0, -0.1, 0])]
third_arm_default_angles = [np.pi / 2, 0, 0, 0, 0, 0, 0, 0, np.pi / 4]
