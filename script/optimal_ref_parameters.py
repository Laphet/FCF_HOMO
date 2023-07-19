import numpy as np
from scipy.optimize import linprog


kvals = np.fromfile("bin/k-maxmin-vals.bin", dtype=np.float64)
print(kvals)
