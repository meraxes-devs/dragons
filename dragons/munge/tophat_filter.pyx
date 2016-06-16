import cython
from libc.math cimport sin, cos, sqrt
import numpy as np
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
def tophat_filter(complex[:, :, :] grid, double side_length, double radius):
    """Apply a tophat filter to a fourier space grid.

    Parameters
    ----------
    grid : 3d ndarray with complex dtype
    side_length : float
    radius : float

    Returns
    -------
    Nothing - the grid is modified **inplace**.

    Raises
    ------
    ValueError
        If the grid is not 3d.

    Notes
    -----
    This function is implemented in order to save memory when filtering large
    grids.  It would be faster to use numpy.mgrid to create a 3d kR array and
    then calculate the filtered fourier space grid directly using numpy.
    """

    if grid.ndim != 3:
        raise ValueError("Grid must be 3d.")

    cdef int dim = grid.shape[0]
    cdef double[:] k = 2.0 * np.pi * \
            np.fft.fftfreq(dim, 1/float(dim)) / side_length
    cdef double[:] k_r = 2.0 * np.pi * \
            np.fft.rfftfreq(dim, 1/float(dim)) / side_length
    cdef int n_k = k.shape[0]
    cdef int n_k_r = k_r.shape[0]
    cdef double k_i, k_j, k_k, kR, val
    cdef int ii, jj, kk

    for ii in range(n_k):
        k_i = k[ii]
        for jj in range(n_k):
            k_j = k[jj]
            for kk in range(n_k_r):
                k_k = k_r[kk]
                kR = sqrt(k_i*k_i + k_j*k_j + k_k*k_k)*radius
                if kR > 0:
                    val = 3.0 * (sin(kR)/(kR**3) - cos(kR)/(kR**2))
                    grid[ii, jj, kk].real = grid[ii, jj, kk].real * val
                    grid[ii, jj, kk].imag = grid[ii, jj, kk].imag * val

