# coding=utf8
from . import meta_model
from . import creature_interface


class MetaCreature(meta_model.MetaModel, creature_interface.CreatureInterface):
    def __init__(self, creature_type="torso", interface_type=None):
        # Base class : meta_model
        meta_model.MetaModel.__init__(self, models=[])

        # Add PyPot Object
        creature_interface.CreatureInterface.__init__(self, interface_type, creature_type=creature_type)
