from . import resources
from . import model
from . import model_config
import os

resource_file = os.path.dirname(resources.__file__)


class creature(model.Model):
    def __init__(self, creature_type):

        if creature_type == "ergo":
            urdf_file = resource_file + "/poppy_ergo.URDF"
            base_link = "base_link"
            last_link_vector = [0, 0.05, 0]
        elif creature_type == "torso":
            urdf_file = resource_file + "/Poppy_Torso.URDF"
            base_link = "chest"
            last_link_vector = [0, 0.1, 0]
        params = model_config.from_urdf_file(urdf_file, base_link, last_link_vector)
        model.Model.__init__(self, params, computation_method="hybrid", simplify=False)
