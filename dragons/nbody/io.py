#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Routines for reading nbody (gbpHalos, gbpTrees etc.) output files."""

from os import path
from os import listdir
import numpy as np
from astropy import log
from tqdm import tqdm

__author__ = 'Simon Mutch'
__email__ = 'smutch.astro@gmail.com'
__version__ = '0.1.0'

catalog_halo_dtype = np.dtype(dict(names=("id_MBP", "M_vir", "n_particles",
                                          "position_COM", "position_MBP",
                                          "velocity_COM", "velocity_MBP",
                                          "R_vir", "R_halo", "R_max", "V_max",
                                          "sigma_v", "spin", "q_triaxial",
                                          "s_triaxial", "shape_eigen_vectors",
                                          "padding"),
                                   formats=('q', 'f8', 'i4', ('f4', 3),
                                            ('f4', 3), ('f4', 3), ('f4', 3),
                                            'f4', 'f4', 'f4', 'f4', 'f4',
                                            ('f4', 3), 'f4', 'f4',
                                            ('f4', (3, 3)), 'S8')),
                              align=True)

catalog_header_dtype = np.dtype(dict(names=("i_file", "N_files",
                                            "N_halos_file", "N_halos_total"),
                                     formats=['i4', ]*4), align=True)


def read_density_grid(fname):

    """Read in a density grid produced by gbpCode.

    *Args*:
        fname : str
            Full path to input grid file.

    *Returns*:
        grid : array
            The density grid.
    """

    log.info("Reading density grid from %s" % fname)

    with open(fname, "rb") as fin:
        # read the header info
        n_cell = np.fromfile(fin, 'i4', 3)
        np.fromfile(fin, 'f8', 3)  # box_size_grid
        n_grids = np.fromfile(fin, 'i4', 1)[0]
        np.fromfile(fin, 'i4', 1)[0]  # ma_scheme

        # read in the identifier
        ident = np.fromfile(fin, 'S32', 1)[0][:10]

        # read in the grid
        grid = np.fromfile(fin, '<f4', n_cell.cumprod()[-1])

        # keep reading grids until we get the density one
        i_grid = 1
        while((ident != 'rho_r_dark') and (i_grid < n_grids)):
            print "Skipping grid '%s'" % ident
            ident = np.fromfile(fin, 'S32', 1)[0]
            grid = np.fromfile(fin, '<f4', n_cell.cumprod()[-1])
            i_grid += 1

        if (ident != 'rho_r_dark'):
            raise KeyError("rho_r_dark grid does not exist in %s" % fname)

    grid.shape = n_cell
    return grid


def read_halo_catalog(catalog_loc):

    """ Read in a halo catalog produced by gbpCode.

    *Args*:
        catalog_loc : str
            Full path to input catalog file or directory

    *Returns*:
        halo : array
            The catalog of halos
    """

    if type(catalog_loc) is str:
        catalog_loc = [catalog_loc, ]

    if path.isdir(catalog_loc[0]):
        dirname = catalog_loc[0]
        catalog_loc = listdir(catalog_loc[0])
        file_num = [int(f.rsplit(".")[-1]) for f in catalog_loc]
        sort_index = np.argsort(file_num)
        catalog_loc = [catalog_loc[i] for i in sort_index]
        catalog_loc = [path.join(dirname, f) for f in catalog_loc]

    n_halos = np.fromfile(catalog_loc[0], catalog_header_dtype,
                          1)[0]["N_halos_total"]
    halo = np.empty(n_halos, dtype=catalog_halo_dtype)
    print "Reading in {:d} halos...".format(n_halos)

    n_halos = 0
    for f in tqdm(catalog_loc):
        with open(f, "rb") as fd:
            n_halos_file = \
                np.fromfile(fd, catalog_header_dtype, 1)[0]["N_halos_file"]
            halo[n_halos:n_halos+n_halos_file] = \
                np.fromfile(fd, catalog_halo_dtype, n_halos_file)
        n_halos += n_halos_file

    return halo[list(catalog_halo_dtype.names[:-1])]
