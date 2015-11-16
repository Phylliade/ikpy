# coding= utf8
from . import resources
from . import model
from . import model_config
import os
import pypot.vrep
import json
resource_file = os.path.dirname(resources.__file__)
wd = os.getcwd()


class creature(model.Model):
    """The creature class"""
    def __init__(self, creature_type, pypot_type=None):

        if creature_type == "ergo":
            urdf_file = resource_file + "/poppy_ergo.URDF"
            base_link = ["base_link"]
            last_link_vector = [0, 0.05, 0]
            if pypot_type == "vrep":
                pypot_config_file = wd + '/../resources/poppy_ergo.json'
                vrep_scene = wd + '/../resources/poppy_ergo.ttt'
        elif creature_type == "torso_left_arm":
            urdf_file = resource_file + "/Poppy_Torso.URDF"
            base_link = ["chest", "l_shoulder_y"]
            last_link_vector = [0, 0.1, 0]
        elif creature_type == "torso_right_arm":
            urdf_file = resource_file + "/Poppy_Torso.URDF"
            base_link = ["chest", "r_shoulder_y"]
            last_link_vector = [0, 0.1, 0]

        if pypot_type == "vrep":
            # Create pypot object
            with open(pypot_config_file, "r+") as f:
                config_data = f.read()

            pypot_config = dict(json.loads(config_data))
            pypot_object = pypot.vrep.from_vrep(pypot_config, scene=vrep_scene)
        else:
            pypot_object = None

        params = model_config.from_urdf_file(urdf_file, base_link, last_link_vector)
        model.Model.__init__(self, params, pypot_object=pypot_object, computation_method="hybrid", simplify=False)
