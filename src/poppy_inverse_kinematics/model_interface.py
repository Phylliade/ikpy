# coding=utf8


class ModelInterface():
    def __init__(self, pypot_object, interface_type, move_duration=None):
        self.pypot_object = pypot_object
        self.interface_type = interface_type
        self.move_duration = move_duration

    def goto_position(self, motor, position):
        if self.interface_type == "vrep":
            motor.goto_position(position, self.move_duration)
