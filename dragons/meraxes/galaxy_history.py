#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Generate the full (first progenitor line) history of a galaxy."""

from ..munge import ndarray_to_dataframe
from .io import read_gals, read_firstprogenitor_indices, read_descendant_indices
from tqdm import tqdm

import numpy as np


def galaxy_history(fname, gal_id, snapshot, future_snapshot=-1, pandas=False, props=None):

    """ Read in the full first progenitor history of a galaxy at a given final
    snapshot.

    Parameters
    ----------
    fname : str
        Full path to input hdf5 master file.

    gal_id : int
        Unique ID of the target galaxy.

    snapshot : int
        Snapshot at which the history is to be traced from.

    props : list
        A list of galaxy properties requested.  (default: All properties)

    future_snapshot: int
        Also read in the future of the galaxy object up to this snapshot.
        (default: -1 [don't read in future])

    pandas : bool
        Return panads dataframe.  (default = False)

    Returns
    -------
    history : ndarray or DataFrame
        The requested first progenitor history.  If future=True then the
        ndarray includes the future of this object.

    merged_snapshot : int
        If `future_snapshot != -1` then the snapshot at which the galaxy
        merged into another is also returned.  If `merged_snapshot = -1`
        then the galaxy remains until `future_snapshot.`
    """

    gals = read_gals(fname, snapshot=snapshot, props=props, pandas=False,
                     quiet=True)

    start_ind = np.where(gals["ID"] == gal_id)[0][0]
    merged_snapshot = -1

    if future_snapshot == -1:
        future_snapshot = snapshot
    history = np.zeros(future_snapshot+1, dtype=gals.dtype)

    history[snapshot] = gals[start_ind]
    ind = read_firstprogenitor_indices(fname, snapshot)[start_ind]

    if ind == -1:
        raise Warning("This galaxy has no progenitors!")

    for snap in tqdm(list(range(snapshot-1, -1, -1))):
        history[snap] = read_gals(fname, snapshot=snap, pandas=False,
                                  quiet=True, props=props, indices=[ind])
        ind = read_firstprogenitor_indices(fname, snap)[ind]
        if ind == -1:
            break

    if future_snapshot != snapshot:
        ind = start_ind
        for snap in tqdm(list(range(snapshot+1, future_snapshot+1))):
            last_ind = ind
            ind = read_descendant_indices(fname, snap-1)[ind]
            if ind == -1:
                break

            if snap < future_snapshot:
                fp = read_firstprogenitor_indices(fname, snap)[ind]
                if fp != last_ind and merged_snapshot == -1:
                    merged_snapshot = snap

            history[snap] = read_gals(fname, snapshot=snap, pandas=False,
                                      quiet=True, props=props, indices=[ind])

    if pandas:
        history = ndarray_to_dataframe(history)

    if future_snapshot == snapshot:
        return history
    else:
        return history, merged_snapshot
