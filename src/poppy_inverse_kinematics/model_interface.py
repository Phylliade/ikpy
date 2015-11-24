# coding=utf8


class ModelInterface():
    def __init__(self, pypot_object, interface_type):
        self.pypot_object = pypot_object
        self.interface_type = interface_type

    def goto_position(self, motor, position):
        if self.pypot_type == "vrep":
            motor.goto_position(position, 2)
