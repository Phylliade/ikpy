import poppy_inverse_kinematics.creature as model_creature
import numpy as np
import poppy_inverse_kinematics.meta_creature as meta_creature
import time

# parameters
activate_follow = True
interface_type = "vrep"
plot = True
waiting_time = 5

# Create creatures
right_arm = model_creature.creature("torso_right_arm")
left_arm = model_creature.creature("torso_left_arm")
torso = meta_creature.MetaCreature(interface_type=interface_type, creature_type="torso")
torso.add_model(right_arm)
torso.add_model(left_arm)

# Init pypot robot
if interface_type == "robot":
    print("Initializing robot")
    for m in torso.pypot_object.motors:
        m.compliant = False
        m.goal_position = 0

    print("Waiting 10 seconds")
    time.sleep(10)


# Set right arm position
target_left = np.array([-0.27, -0.2, 0.5])
# target_left = np.array([-0.1, -0.4, 0.1])
left_arm.target = target_left
left_arm.goto_target()

# The left arm is now compliant, so it can be moved
left_arm.set_compliance(compliance=True)

# Choose right arm target
if activate_follow:
    try:
        while True:
            print("Waiting %s seconds" % waiting_time)
            time.sleep(waiting_time)
            left_arm.sync_current_joints()
            target_right = left_arm.forward_kinematic() + np.array([0.3, 0, 0])
            right_arm.target = target_right
            right_arm.goto_target()
    except KeyboardInterrupt:
        # Plot result
        if plot:
            torso.plot_meta_model()

        if interface_type == "robot":
            torso.pypot_object.close()
