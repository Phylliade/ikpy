import pytest


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
