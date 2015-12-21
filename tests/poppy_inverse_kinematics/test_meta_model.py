import unittest
import poppy_inverse_kinematics.meta_model as meta_model


class TestMetaModel(unittest.TestCase):
    def test_MetaModel(self):
        meta_model_object = meta_model.MetaModel()


if __name__ == '__main__':
    unittest.main(verbosity=2)
