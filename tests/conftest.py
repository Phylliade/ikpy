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
def baxter_urdf():
    return "../resources/baxter/baxter.urdf"


@pytest.fixture
def baxter_left_arm():
    with open("../resources/baxter/baxter_left_arm_elements.json", "r") as fd:
        baxter_left_arm_elements = json.load(fd)

    baxter_left_arm_chain = Chain.from_urdf_file(
        "../resources/baxter/baxter.urdf",
        base_elements=baxter_left_arm_elements,
        last_link_vector=[0, 0.18, 0],
        active_links_mask=3 * [False] + 10 * [True],
        symbolic=False)
    return baxter_left_arm_chain
