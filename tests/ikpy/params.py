import sys

resources_path = "../../resources"
interactive = True
try:
    if sys.argv[1] == "--no-interactive":
        interactive = False
except:
    pass
