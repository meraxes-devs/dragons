#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Generate the full (first progenitor line) history of a galaxy."""

from ..munge import ndarray_to_dataframe
from io import read_gals, read_firstprogenitor_indices
from astropy.utils.console import ProgressBar

import numpy as np

__author__ = 'Simon Mutch'
__email__ = 'smutch.astro@gmail.com'
__version__ = '0.1.1'


def galaxy_history(fname, gal_id, last_snapshot, pandas=False, props=None):

    """ Read in the full first progenitor history of a galaxy at a given final
    snapshot.

    *Args*:
        fname (str): Full path to input hdf5 master file.

        gal_id (int): Unique ID of the target galaxy.

        last_snapshot (int): Last snapshot (lowest redshift) at which the
                             history is to be traced from.

        props (list): A list of galaxy properties requested.
                      (default: All properties)

    *Kwargs*:
        pandas (bool): Return panads dataframe.
                       (default = False)

    *Returns*:
        history (ndarray or DataFrame): The requested first progenitor history.
    """

    gals = read_gals(fname, snapshot=last_snapshot, props=props, pandas=False, quiet=True)

    ind = np.where(gals["ID"] == gal_id)[0][0]

    history = np.zeros(last_snapshot+1, dtype=gals.dtype)

    history[last_snapshot] = gals[ind]
    ind = read_firstprogenitor_indices(fname, last_snapshot)[ind]

    if ind == -1:
        raise IndexError("This galaxy has no progenitors!")

    with ProgressBar(last_snapshot) as bar:
        for snap in xrange(last_snapshot-1, -1, -1):
            history[snap] = read_gals(fname, snapshot=snap, pandas=False,
                                      quiet=True, props=props, indices=[ind])
            ind = read_firstprogenitor_indices(fname, snap)[ind]
            if ind == -1:
                break
            bar.update()

    if pandas:
        history = ndarray_to_dataframe(history)

    return history
