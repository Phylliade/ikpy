robot = "ergo"
representation = "rpy"
model_type = "URDF"


if robot == "ergo":
    urdf_file = "../resources/poppy_ergo.URDF"
    base_link = "base_link"
    last_link_vector = None
    vrep_scene = '../resources/poppy_ergo.ttt'
    robot_json = "../resources/poppy_ergo.json"
elif robot == "torso":
    urdf_file = "../resources/Poppy_Torso.URDF"
    base_link = "chest"
    last_link_vector = [0, 0.1, 0]
    vrep_scene = '../resources/poppy_torso.ttt'
    robot_json = "../resources/poppy_torso.json"
