#cython: boundscheck=False
#cython: wraparound=False
#cython: language_level=3

from __future__ import division

import numpy as np
from tqdm import tqdm

cimport numpy as np

import logging

logger = logging.getLogger(__name__)
logger.setLevel('WARNING')

def regrid(np.ndarray[np.float32_t, ndim=3] old_grid not None,
           int n_cell):

    """ Downgrade the resolution of a 3 dimensional grid.

    Parameters
    ----------
    old_grid (np.ndarray[float32, ndim=3]) :  Grid to be resampled

    n_cell (int) :  n cells per dimension of new grid

    Returns
    -------
    New, degraded resolution grid.
    """

    if n_cell > old_grid.shape[0]:
        logger.warning("regrid function is untested for n_cell > old_grid.shape[0]")

    cdef int old_dim = old_grid.shape[0]
    cdef float resample_factor = float(n_cell/old_dim)

    cdef np.ndarray[np.float32_t, ndim=3] new_grid = np.zeros([n_cell,n_cell,n_cell], np.float32)

    cdef unsigned int i,j,k, rsi, rsj, rsk
    for i in tqdm(xrange(old_dim)):
        for j in xrange(old_dim):
            for k in xrange(old_dim):
                rsi = int(i*resample_factor)
                rsj = int(j*resample_factor)
                rsk = int(k*resample_factor)
                new_grid[rsi, rsj, rsk] += old_grid[i,j,k]

    return new_grid
