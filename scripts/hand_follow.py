import time
import numpy as np
from poppy.creatures import PoppyTorso


poppy = PoppyTorso(simulator='vrep')

delay_time = 1
target_delta = np.array([-0.15, 0, 0])

# Initialize the robot
for m in poppy.motors:
    m.goto_position(0, 2)

# Left arm is compliant, right arm is active
for m in poppy.l_arm:
            m.compliant = False

for m in poppy.r_arm:
    m.compliant = False

# The torso itself must not be compliant
for m in poppy.torso:
    m.compliant = False


def follow_hand(poppy, delta):
    """Tell the right hand to follow the left hand"""
    right_arm_position = poppy.l_arm_chain.end_effector + delta
    poppy.r_arm_chain.goto(right_arm_position, 0.5, wait=True)

poppy.l_arm_chain.goto(poppy.l_arm_chain.end_effector - target_delta, 5.5, wait=True)

try:
    while True:
        follow_hand(poppy, target_delta)
        time.sleep(delay_time)

except KeyboardInterrupt:
    poppy.close()
