#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A collection of functions for doing common processing tasks."""

import re
import numpy as np
from astropy import log
from astroML.density_estimation import scotts_bin_width, freedman_bin_width,\
    knuth_bin_width, bayesian_blocks
from pandas import DataFrame
from scipy.stats import describe as sp_describe

__author__ = 'Simon Mutch'
__email__ = 'smutch.astro@gmail.com'
__version__ = '0.1.0'


def pretty_print_dict(d, fmtlen=30):

    """Pretty print a dictionary, dealing with recursion.

    *Args*:
        d : dict
            the dictionary to print

        fmtlen : int
            maximum length of dictionary key for print formatting
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
        arr : ndarray
            Numpy ndarray

    *Kwargs*:
        drop_vectors : bool
            only include single value datatypes in output DataFrame

    *Returns*:
        df : DataFrame
            Pandas DataFrame
    """

    # Get a list of all of the columns which are 1D
    names = []
    for k, v in arr.dtype.fields.iteritems():
        if len(v[0].shape) == 0:
            names.append(k)

    # Create a new dataframe with these columns
    df = DataFrame(arr[names])

    if not drop_vectors:
        # Loop through each N(>1)D property and append each dimension as
        # its own column in the dataframe
        for k, v in arr.dtype.fields.iteritems():
            if len(v[0].shape) != 0:
                for i in range(v[0].shape[0]):
                    df[k+'_%d' % i] = arr[k][:, i]

    return df


def mass_function(mass, volume, bins, range=None, poisson_uncert=False,
                  return_edges=False, **kwargs):

    """Generate a mass function.

    *Args*:
        mass : array
            an array of 'masses'

        volume : float
            volume of simulation cube/subset

        bins : int or list or str
            If bins is a string, then it must be one of:
                | 'blocks'   : use bayesian blocks for dynamic bin widths
                | 'knuth'    : use Knuth's rule to determine bins
                | 'scott'    : use Scott's rule to determine bins
                | 'freedman' : use the Freedman-diaconis rule to determine bins

    *Kwargs*:
        range : len=2 list or array
            range of data to be used for mass function

        poisson_uncert : bool
            return poisson uncertainties in output array (default: False)

        return_edges : bool
            return the bin_edges (default: False)

        **kwargs
            passed to np.histogram call

    *Returns*:
        array of [bin centers, mass function vals]

        If poisson_uncert=True then array has 3rd column with uncertainties.

        If return_edges=True then the bin edges are also returned.

    *Notes*:
        The code to generate the bin_widths is taken from astroML.hist

    """

    if "normed" in kwargs:
        kwargs["normed"] = False
        log.warn("Turned off normed kwarg in mass_function()")

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
    if poisson_uncert:
        uncert = np.sqrt(vals.astype(float))

    vals = vals.astype(float) / (volume * width)

    if not poisson_uncert:
        mf = np.dstack((centers, vals)).squeeze()
    else:
        uncert /= (volume * width)
        mf = np.dstack((centers, vals, uncert)).squeeze()

    if not return_edges:
        return mf
    else:
        return mf, edges


def edges_to_centers(edges, width=False):

    """Convert **evenly spaced** bin edges to centers.

    *Args*:
        edges : ndarray
            bin edges

    *Kwargs*:
        width : bool
            also return the bin width

    *Returns*:
        centers : ndarray
            bin centers (size = edges.size-1)

        bin_width : float
            only returned if width = True

    """

    bin_width = edges[1] - edges[0]
    radius = bin_width * 0.5
    centers = edges[:-1] + radius

    if width:
        return centers, bin_width
    else:
        return centers


def describe(arr, **kwargs):

    """Run scipy.stats.describe and produce legible output.

    *Args*:
        arr : ndarray
            Numpy ndarray

    *Kwargs*:
        passed to scipy.stats.describe

    *Returns*:
        output of scipy.stats.describe
    """

    stats = sp_describe(arr)

    print("{:15s} : {:g}".format("size", stats[0]))
    print("{:15s} : {:g}".format("min", stats[1][0]))
    print("{:15s} : {:g}".format("max", stats[1][1]))
    print("{:15s} : {:g}".format("mean", stats[2]))
    print("{:15s} : {:g}".format("unbiased var", stats[3]))
    print("{:15s} : {:g}".format("biased skew", stats[4]))
    print("{:15s} : {:g}".format("biased kurt", stats[5]))

    return stats
