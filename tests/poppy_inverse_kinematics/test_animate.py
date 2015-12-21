import unittest
import poppy_inverse_kinematics.robot_utils
import poppy_inverse_kinematics.model
import poppy_inverse_kinematics.model_config
import poppy_inverse_kinematics.creature
import poppy_inverse_kinematics.plot_utils as plot_utils
import numpy as np
import matplotlib.animation


class TestModel(unittest.TestCase):
    def test_animate(self):
        creature = poppy_inverse_kinematics.creature.creature("torso_right_arm")
        arm_length = creature.arm_length
        t = np.arange(0, np.pi / 2, np.pi / 50)
        x = np.sin(t**2) * arm_length / 3
        y = np.sin(t) * arm_length / 3
        z = -np.sinh(t) * arm_length / 3

        x2 = np.sqrt(x)
        y2 = y**2
        z2 = z

        # Génération de l'animation
        animation = creature.animate_model(x, y, z)

        # Définition d'un writer pour enregistrer une video depuis l'animation
        Writer = matplotlib.animation.writers['ffmpeg']
        animation_writer = Writer(fps=30, bitrate=3600)

if __name__ == '__main__':
    unittest.main(verbosity=2)
