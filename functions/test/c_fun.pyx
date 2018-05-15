from cython.view cimport array as cvarray
import numpy as np
cimport numpy as cnp
cimport cython
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function

def main(double [:] arr_view_1, double [:] arr_view_2):

    cdef object[double, ndim = 1] out_arr = np.zeros(arr_view_1.shape[0], dtype = np.float64)
    cdef double [:] out_view = out_arr
    cdef int i

    for i in xrange(0, arr_view_1.shape[0]):

		# add views
        out_view[i] = arr_view_1[i] + arr_view_2[i]

	# out array
    return out_arr

# np.asarray(out_view, dtype = np.float64)
