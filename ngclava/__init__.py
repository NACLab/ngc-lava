from ngclearn import numpy as _numpy

#Verify that the base version of numpy is being used
if _numpy.__name__ != "numpy":
    raise RuntimeError(f"In order to build lava components the base version of numpy must be used: Current version {_numpy.__name__}.\n"
                       f"Using the NGC-Learn configuration file this can be done by setting the flag \"use_base_numpy\" to true in the "
                       f"\"packages\" section of the file")
del _numpy

from .lavaContext import LavaContext