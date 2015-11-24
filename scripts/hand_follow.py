import poppy_inverse_kinematics.creature as model_creature
import numpy as np
import poppy_inverse_kinematics.meta_creature as meta_creature
import time

# parameters
activate_follow = True
waiting_time = 5
exp_type = "simulation"
target_delta = np.array([-0.3, 0, 0])

if exp_type == "simulation":
    manual_move = False
    interface_type = "vrep"
    plot = True
elif exp_type == "real":
    manual_move = True
    interface_type = "robot"
    plot = False


def exit_script(model):
    if plot:
        torso.plot_meta_model()

    if interface_type == "robot":
        torso.pypot_object.close()


def follow_hand(left_arm, right_arm):
    left_arm.sync_current_joints()
    target_right = left_arm.forward_kinematic() + target_delta
    right_arm.target = target_right
    right_arm.goto_target()

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


# Set left arm position
target_left = np.array([0.27, -0.2, 0.5])
# target_left = np.array([-0.1, -0.4, 0.1])
left_arm.target = target_left
left_arm.goto_target()

if manual_move:
    # The left arm is now compliant, so it can be moved
    left_arm.set_compliance(compliance=True)

# Choose right arm target
if activate_follow:
    if manual_move:
        try:
            while True:
                print("Waiting %s seconds" % waiting_time)
                time.sleep(waiting_time)
                follow_hand(left_arm, right_arm)
        except KeyboardInterrupt:
            # Plot result
            exit_script(torso)
    else:
        follow_hand(left_arm, right_arm)
        exit_script(torso)
