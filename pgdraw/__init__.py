import numpy as np
import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()}, inplace=True)
from pgdraw.pgdraw import pgdraw
