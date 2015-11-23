# coding= utf8
from . import resources
from . import model
from . import model_config
from . import creature_interface
import os
resource_file = os.path.dirname(resources.__file__)
wd = os.getcwd()


class creature(model.Model, creature_interface.CreatureInterface):
    """The creature class"""
    def __init__(self, creature_type, interface_type=None):

        if creature_type == "ergo":
            urdf_file = resource_file + "/poppy_ergo.URDF"
            base_link = ["base_link"]
            last_link_vector = [0, 0.05, 0]
            motor_config_file = wd + '/../resources/poppy_ergo.json'
        elif creature_type == "torso_left_arm":
            urdf_file = resource_file + "/Poppy_Torso.URDF"
            base_link = ["base", "abs_z", "spine", "bust_y", "bust_motors", "bust_x", "chest", "l_shoulder_y"]
            last_link_vector = [0, 0.25, 0]
            motor_config_file = wd + '/../resources/poppy_torso.json'
        elif creature_type == "torso_right_arm":
            urdf_file = resource_file + "/Poppy_Torso.URDF"
            base_link = ["base", "abs_z", "spine", "bust_y", "bust_motors", "bust_x", "chest", "r_shoulder_y"]
            last_link_vector = [0, 0.25, 0]
            motor_config_file = wd + '/../resources/poppy_torso.json'

        params = model_config.from_urdf_file(urdf_file, base_link, last_link_vector, motor_config_file=motor_config_file)
        model.Model.__init__(self, params, computation_method="hybrid", simplify=False)

        # Add PyPot object
        creature_interface.CreatureInterface.__init__(self, interface_type=interface_type, creature_type=creature_type)
