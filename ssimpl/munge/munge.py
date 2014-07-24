#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A collection of functions for doing common processing tasks."""

import re
import numpy as np
from astropy import log
from astroML.density_estimation import scotts_bin_width, freedman_bin_width,\
    knuth_bin_width, bayesian_blocks
import pandas as pd

__author__ = 'Simon Mutch'
__email__ = 'smutch.astro@gmail.com'
__version__ = '0.1.0'


def pretty_print_dict(d, fmtlen=30):

    """Pretty print a dictionary, dealing with recursion.

    *Args*:
        d (dict): The dictionary to print

        fmtlen (int): maximum length of dictionary key for print formatting
    """

    fmtstr = "%%%ds :" % fmtlen
    fmtstr_title = "\n%%%ds\n%%%ds" % (fmtlen, fmtlen)

    for k, v in d.iteritems():
        if isinstance(v, dict):
            print fmtstr_title % (k.upper(), re.sub('\S', '-', k))
            pretty_print_dict(v)
        else:
            print fmtstr % k, v


def ndarray_to_dataframe(arr, drop_vectors=False):

    """Convert numpy ndarray to a pandas DataFrame, dealing with N(>1)
    dimensional datatypes.

    *Args*:
        arr (ndarray): Numpy ndarray

    *Kwargs*:
        drop_vectors (bool): only include single value datatypes in output
                             DataFrame

    *Returns*:
        df (DataFrame): Pandas DataFrame
    """

    # Get a list of all of the columns which are 1D
    names = []
    for k, v in arr.dtype.fields.iteritems():
        if len(v[0].shape) == 0:
            names.append(k)

    # Create a new dataframe with these columns
    df = pd.DataFrame(arr[names])

    if not drop_vectors:
        # Loop through each N(>1)D property and append each dimension as
        # its own column in the dataframe
        for k, v in arr.dtype.fields.iteritems():
            if len(v[0].shape) != 0:
                for i in range(v[0].shape[0]):
                    df[k+'_%d' % i] = arr[k][:, i]

    return df


def mass_function(mass, volume, bins,
                  range=None, return_edges=False, **kwargs):

    """Generate a mass function.

    *Args*:
        mass (array):  an array of 'masses'

        volume (float):  volume of simulation cube/subset

        bins (int or list or str):
            If bins is a string, then it must be one of:
                | 'blocks'   : use bayesian blocks for dynamic bin widths
                | 'knuth'    : use Knuth's rule to determine bins
                | 'scott'    : use Scott's rule to determine bins
                | 'freedman' : use the Freedman-diaconis rule to determine bins

    *Kwargs*:
        range (len=2 list or array): range of data to be used for mass function

        return_edges (bool): return the bin_edges (default: False)

        \*\*kwargs: passed to np.histogram call

    *Returns*:
        array of [bin centers, mass function vals]

        If return_edges=True then the bin edges are also returned.

    *Notes*:
        The code to generate the bin_widths is taken from astroML.hist

    """

    if "normed" in kwargs:
        kwargs["normed"] = False
        log.warn("Turned of normed kwarg in mass_function()")

    if (range is not None and (bins in ['blocks',
                                        'knuth', 'knuths',
                                        'scott', 'scotts',
                                        'freedman', 'freedmans'])):
        mass = mass[(mass >= range[0]) & (mass <= range[1])]

    if isinstance(bins, str):
        log.info("Calculating bin widths using `%s' method..." % bins)
        if bins in ['blocks']:
            bins = bayesian_blocks(mass)
        elif bins in ['knuth', 'knuths']:
            dm, bins = knuth_bin_width(mass, True)
        elif bins in ['scott', 'scotts']:
            dm, bins = scotts_bin_width(mass, True)
        elif bins in ['freedman', 'freedmans']:
            dm, bins = freedman_bin_width(mass, True)
        else:
            raise ValueError("unrecognized bin code: '%s'" % bins)
        log.info("...done")

    vals, edges = np.histogram(mass, bins, range, **kwargs)
    width = edges[1]-edges[0]
    radius = width/2.0
    centers = edges[:-1]+radius
    vals = vals.astype(float) / (volume * width)

    mf = np.dstack((centers, vals)).squeeze()

    if not return_edges:
        return mf
    else:
        return mf, edges
