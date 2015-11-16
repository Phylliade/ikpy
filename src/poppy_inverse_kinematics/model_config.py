from . import robot_utils


class model_config():
    """Configuration of a model"""
    def __init__(self, links, representation="euler", model_type="custom", name="robot"):
        self.parameters = links
        self.joints_number = len(links)
        self.representation = representation
        self.model_type = model_type
        self.name = name

    def __repr__(self):
        return("Configuration of robot : {0}".format(self.name))


def from_urdf_file(urdf_file, base_elements=["base_link"], last_link_vector=None, base_elements_type="joint"):
    urdf_params = robot_utils.get_urdf_parameters(urdf_file, base_elements=base_elements, last_link_vector=last_link_vector, base_elements_type=base_elements_type)
    return model_config(urdf_params, representation="rpy", model_type="URDF")
