#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Routines for reading nbody (gbpHalos, gbpTrees etc.) output files."""

import numpy as np
import h5py as h5
from astropy import log

__author__ = 'Simon Mutch'
__email__ = 'smutch.astro@gmail.com'
__version__ = '0.1.0'

def read_density_grid(fname):

    """ Read in a density grid produced by gbpCode.

    *Args*:
        fname (str): Full path to input grid file.

    *Returns*:
        grid (array): The density grid.
    """

    with open(fname, "rb") as fin:
        # read the header info
        n_cell = np.fromfile(fin, 'i4', 3)
        box_size_grid = np.fromfile(fin, 'f8', 3)
        n_grids = np.fromfile(fin, 'i4', 1)[0]
        ma_scheme = np.fromfile(fin, 'i4', 1)[0]

        # read in the identifier
        ident = np.fromfile(fin, 'S32', 1)[0]

        # read in the grid
        grid = np.fromfile(fin, '<f4', n_cell.cumprod()[-1])

        # keep reading grids until we get the density one
        i_grid = 1
        while((ident != 'rho_r_dark') and (i_grid<n_grids)):
            ident = np.fromfile(fin, 'S32', 1)[0]
            grid = np.fromfile(fin, '<f4', n_cell.cumprod()[-1])

        if (ident != 'rho_r_dark'):
            raise KeyError("rho_r_dark grid does not exist in %s" % fname)

    grid.shape = n_cell
    return grid
