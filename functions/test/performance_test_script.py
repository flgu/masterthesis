import numpy as np

import c_fun

# define 1 dim numpy array
arr = np.array([1,2,3,4,5,6], dtype = np.double)

out = c_fun.main( arr, arr )