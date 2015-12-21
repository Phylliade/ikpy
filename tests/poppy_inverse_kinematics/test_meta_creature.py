import unittest
import poppy_inverse_kinematics.meta_creature as meta_creature


class TestMetaCreature(unittest.TestCase):
    def test_MetaModel(self):
        meta_creature_object = meta_creature.MetaCreature()


if __name__ == '__main__':
    unittest.main(verbosity=2)
