from ngclearn import numpy as _numpy
from ngcsimlib.logger import warn

_can_use_lava = True
#Verify that the base version of numpy is being used
if _numpy.__name__ != "numpy":
    warn(f"In order to build lava components the base version of numpy must be used: Current version {_numpy.__name__}.\n"
                       f"Using the NGC-Learn configuration file this can be done by setting the flag \"use_base_numpy\" to true in the "
                       f"\"packages\" section of the file")
    _can_use_lava = False

def lava_compatible_env():
    return _can_use_lava

del _numpy

try:
    import lava
    del lava
except:
    raise RuntimeError("Unable to import lava, please make sure it is installed and can be imported with `import lava`")

from .lavaContext import LavaContext