import sys

resources_path = "../resources"

interactive = False
try:
    if sys.argv[1] == "--interactive":
        interactive = True
except KeyError:
    pass
