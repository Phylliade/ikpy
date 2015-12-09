import unittest
from poppy_inverse_kinematics import chain
import test_resources


class TestChain(unittest.TestCase):
    def test_chain(self):
        chain.Chain.from_urdf_file(test_resources.resources_path + "/poppy_ergo.URDF")


if __name__ == '__main__':
    unittest.main(verbosity=2)
