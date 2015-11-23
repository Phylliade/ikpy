import poppy_inverse_kinematics.creature as model_creature
import numpy as np
import poppy_inverse_kinematics.meta_creature as meta_creature
import time

# parameters
activate_follow = True
interface_type = "robot"
plot = False

# Create creatures
right_arm = model_creature.creature("torso_right_arm")
left_arm = model_creature.creature("torso_left_arm")
torso = meta_creature.MetaCreature(interface_type=interface_type, creature_type="torso")
torso.add_model(right_arm)
torso.add_model(left_arm)

# Init pypot robot
if interface_type == "robot":
    for m in torso.pypot_object.motors:
        m.compliant = False
        m.goal_position = 0
    time.sleep(10)


# Set right arm position
target_right = np.array([-0.27, -0.2, 0.5])
# target_right = np.array([-0.1, -0.4, 0.1])
right_arm.target = target_right
right_arm.goto_target()

# Choose left arm target
if activate_follow:
    target_left = right_arm.forward_kinematic() + np.array([0.3, 0, 0])
    left_arm.target = target_left
    left_arm.goto_target()

# Plot result
if plot:
    torso.plot_meta_model()

if interface_type == "robot":
    torso.pypot_object.close()
