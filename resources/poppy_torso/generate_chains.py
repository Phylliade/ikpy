# IKPy imports
from ikpy.chain import Chain


def get_torso_right_arm():
    chain1 = Chain.from_urdf_file(
        "./poppy_torso.URDF",
        base_elements=[
            "base", "abs_z", "spine", "bust_y", "bust_motors", "bust_x",
            "chest", "r_shoulder_y"
        ],
        last_link_vector=[0, 0.18, 0],
        active_links_mask=[
            False, False, False, False, True, True, True, True, True
        ],
        name="poppy_torso_right_arm"
    )
    return chain1


def get_torso_left_arm():
    return Chain.from_urdf_file(
        "./poppy_torso.URDF",
        base_elements=[
            "base", "abs_z", "spine", "bust_y", "bust_motors", "bust_x",
            "chest", "l_shoulder_y"
        ],
        last_link_vector=[0, 0.18, 0],
        active_links_mask=[
            False, False, False, False, True, True, True, True, True
        ],
        name="poppy_torso_left_arm"
    )


if __name__ == "__main__":
    left_arm = get_torso_left_arm()
    left_arm.to_json_file(force=True)
    right_arm = get_torso_right_arm()
    right_arm.to_json_file(force=True)
