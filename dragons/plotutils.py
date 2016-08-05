"""Small plotting-related utility functions.

Three new matplotlib styles are added to the user library upon import:

- dragons
- talk
- white_background

These can be combined as required.
"""


from __future__ import division
import numpy as np
from matplotlib import style
from pkg_resources import resource_filename
from scipy import optimize as so


# Add custom styles
style.core.USER_LIBRARY_PATHS.append(resource_filename(__name__,
                                                       'stylelib'))
style.reload_library()


def _find_confidence_interval(x, pdf, confidence_level):
    return pdf[pdf > x].sum() - confidence_level


def density_contour(xdata, ydata, bins, ax, label=True,
                    clabel_kwargs={}, **contour_kwargs):
    """ Create a density contour plot.

    Code modified from:
    https://gist.github.com/adrn/3993992#file-density_contour-py

    Parameters
    ----------
    xdata : ndarray

    ydata : ndarray

    bins : int or list
        Number of bins [nbins_x, nbins_y]. If int then
        nbins_x=nbins_y=nbins.

    ax : matplotlib.axes.AxesSubplot
        Axis to draw contours on

    label : bool
        Draw labels on the contours? (default: True)

    clabel_kwargs : dict
        kwargs to be passed to pyplot.clabel() (default: {})

    \*\*contour_kwargs : dict
        kwargs to be passed to pyplot.contour()

    Returns
    -------
    contour : matplotlib.contour.QuadContourSet
    """

    if type(bins) is int:
        nbins_x = nbins_y = bins
    else:
        nbins_x = bins[0]
        nbins_y = bins[1]

    H, xedges, yedges = np.histogram2d(xdata, ydata, bins=bins,
                                       normed=True)
    x_bin_sizes = (xedges[1:] - xedges[:-1]).reshape((1, nbins_x))
    y_bin_sizes = (yedges[1:] - yedges[:-1]).reshape((nbins_y, 1))

    pdf = (H*(x_bin_sizes*y_bin_sizes))

    one_sigma = so.brentq(_find_confidence_interval, 0., 1.,
                          args=(pdf, 0.39346934))
    two_sigma = so.brentq(_find_confidence_interval, 0., 1.,
                          args=(pdf, 0.864664717))
    three_sigma = so.brentq(_find_confidence_interval, 0., 1.,
                            args=(pdf, 0.988891003))
    levels = [three_sigma, two_sigma, one_sigma]

    X, Y = 0.5*(xedges[1:]+xedges[:-1]), 0.5*(yedges[1:]+yedges[:-1])
    Z = pdf.T

    contour = ax.contour(X, Y, Z, levels=levels, origin="lower",
                         **contour_kwargs)

    if label:
        lim = ax.axis()

        fmt = {}
        strs = ['39%', '86%', '99%'][::-1]
        for l, s in zip(contour.levels, strs):
            fmt[l] = s

        ax.clabel(contour, contour.levels, fmt=fmt, **clabel_kwargs)

        ax.axis(lim)

    return contour
