# coding=utf8
import json
import os
wd = os.getcwd()


class CreatureInterface():
    """Class implementing the PyPot interface for creature and meta_creature"""
    def __init__(self, interface_type="vrep", creature_type="torso", move_duration=None):
        if creature_type == "torso" or creature_type == "torso_right_arm" or creature_type == "torso_left_arm":
            pypot_config_file = wd + '/../resources/poppy_torso.json'
            if move_duration is None:
                self.move_duration = 2
            if interface_type == "vrep":
                vrep_scene = wd + '/../resources/poppy_torso.ttt'
        elif creature_type == "ergo":
            pypot_config_file = wd + '/../resources/poppy_ergo.json'
            if move_duration is None:
                self.move_duration = 2
            if interface_type == "vrep":
                vrep_scene = wd + '/../resources/poppy_ergo.ttt'

        if interface_type is not None:
            # Create pypot object
            with open(pypot_config_file, "r+") as f:
                config_data = f.read()
            pypot_config = dict(json.loads(config_data))

            if interface_type == "vrep":
                # VREP interface
                # Import pypot only if necessary
                import pypot.vrep
                self.pypot_object = pypot.vrep.from_vrep(pypot_config, scene=vrep_scene)

            elif interface_type == "robot":
                # Standard PyPot interface
                import pypot.robot
                self.pypot_object = pypot.robot.from_config(pypot_config)

        else:
            # Default case : no object
            self.pypot_object = None
