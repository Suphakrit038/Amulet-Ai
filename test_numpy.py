import os
import sys

# Debug logging for numpy import issue
print("DEBUG: Current working directory:", os.getcwd())
print("DEBUG: Python executable:", sys.executable)
print("DEBUG: PYTHONPATH:", os.environ.get('PYTHONPATH', 'Not set'))
print("DEBUG: sys.path[:3]:", sys.path[:3])

try:
    import numpy as np
    print("SUCCESS: NumPy imported successfully")
    print("NumPy location:", np.__file__)
    print("NumPy version:", np.__version__)
except ImportError as e:
    print("ERROR: Failed to import NumPy:", str(e))