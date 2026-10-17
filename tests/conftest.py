import os
import sys

# The compiled module and the scripts are in python/.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
