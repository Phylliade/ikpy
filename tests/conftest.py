import pytest
import json

# IKPy imports
from ikpy.chain import Chain


@pytest.fixture
def resources_path():
    return "../resources"


def pytest_addoption(parser):
    parser.addoption(
        "--interactive", action="store_true", help="activate interactive mode"
    )


@pytest.fixture
def interactive(request):
    return request.config.getoption("--interactive")


@pytest.fixture
def poppy_torso_urdf():
    return "../resources/poppy_torso/poppy_torso.URDF"


@pytest.fixture
def baxter_urdf():
    return "../resources/baxter/baxter.urdf"


@pytest.fixture
def poppy_ergo_urdf():
    return "../resources/poppy_ergo.URDF"


@pytest.fixture
def baxter_left_arm():
    baxter_left_arm_chain = Chain.from_json_file("../resources/baxter/baxter_left_arm.json")
    return baxter_left_arm_chain


@pytest.fixture
def torso_right_arm():
    chain1 = Chain.from_urdf_file(
        "../resources/poppy_torso/poppy_torso.URDF",
        base_elements=[
            "base", "abs_z", "spine", "bust_y", "bust_motors", "bust_x",
            "chest", "r_shoulder_y"
        ],
        last_link_vector=[0, 0.18, 0],
        active_links_mask=[
            False, False, False, False, True, True, True, True, True
        ],
        name="poppy_torso_right_arm"
    )
    return chain1


@pytest.fixture
def torso_left_arm():
    return Chain.from_urdf_file(
        "../resources/poppy_torso/poppy_torso.URDF",
        base_elements=[
            "base", "abs_z", "spine", "bust_y", "bust_motors", "bust_x",
            "chest", "l_shoulder_y"
        ],
        last_link_vector=[0, 0.18, 0],
        active_links_mask=[
            False, False, False, False, True, True, True, True, True
        ],
        name="poppy_torso_left_arm"
    )
