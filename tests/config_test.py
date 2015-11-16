robot = "torso_right_arm"
representation = "rpy"
model_type = "URDF"


if robot == "ergo":
    urdf_file = "../resources/poppy_ergo.URDF"
    base_elements = ["base_link"]
    base_elements_type = "joint"
    last_link_vector = [0.05, 0, 0]
    vrep_scene = '../resources/poppy_ergo.ttt'
    robot_json = "../resources/poppy_ergo.json"
    predicted_config = [([-0.000819116943949499, 0.0, 0.0395000000000191], [0.0, 0.0, 1.5707963267949], [0.0, 0.0, -1.0]), ([0.0, 0.0, 0.03], [1.5707963267949, 0.0, 0.0], [-1.0, 0.0, 0.0]), ([0.0, 0.07, 0.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]), ([0.0, 0.03715, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]), ([0.0, 0.0258500000000001, 0.0702000000000001], [1.5707963267949, 0.0, 0.0], [-1.0, 0.0, 0.0]), ([0.0, 0.07, 0.0], [0.0, 0.0, -9.36192041654558e-16], [-1.0, 0.0, 0.0]), ([0.05, 0, 0], [0, 0, 0], [0, 0, 0])]

elif robot == "torso_right_arm":
    urdf_file = "../resources/Poppy_Torso.URDF"
    base_elements = ["chest", "r_shoulder_y"]
    base_elements_type = "joint"
    last_link_vector = [0, 0.1, 0]
    vrep_scene = '../resources/poppy_torso.ttt'
    robot_json = "../resources/poppy_torso.json"
