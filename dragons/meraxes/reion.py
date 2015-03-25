#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Routines for reionisation related calculations."""

import numpy as np
from astropy import cosmology
from astropy import units as U
from astropy import constants as C
from scipy import integrate

from .io import read_input_params, read_snaplist, read_global_xH, read_grid

__author__ = 'Simon Mutch'
__email__ = 'smutch.astro@gmail.com'
__version__ = '0.1.1'


def electron_optical_depth(fname):
    """Calculate the electron Thomson scattering optical depth from a Meraxes +
    21cmFAST run."""

    # read in the model run parameters and set up the cosmology
    run_params = read_input_params(fname)
    cosmo = cosmology.FlatLambdaCDM(H0=run_params['Hubble_h']*100.,
                                    Om0=run_params['OmegaM'],
                                    Ob0=run_params['OmegaM']
                                    * run_params['BaryonFrac'])

    # define the necessary constants
    thomson_cross_section = 6.652e-25 * U.cm * U.cm
    density_H = 1.88e-7 * cosmo.Ob0 * cosmo.h**2 / 0.022 * U.cm**-3
    density_He = 0.19e-7 * cosmo.Ob0 * cosmo.h**2 / 0.022 * U.cm**-3

    cosmo_factor = lambda z: C.c * (1+z)**2 \
        / cosmo.H(z) * thomson_cross_section

    # read in the model run data
    snaps, z_list, _ = read_snaplist(fname)
    xHII = 1.0 - read_global_xH(fname, snaps)

    # reweight the ionised fraction by mass
    sel = ~(np.isclose(xHII, 0) | np.isclose(xHII, 1))
    for ii, snap in enumerate(snaps[sel]):
        deltax = read_grid(fname, snap, 'deltax', h=run_params['Hubble_h'])
        xHII_grid = 1.0 - read_grid(fname, snap, 'xH',
                                    h=run_params['Hubble_h'])
        xHII[sel][ii] = np.mean(deltax * xHII_grid)

    def d_te_postsim(z):
        """This is d/dz scattering depth for redshifts greater than the final
        redshift of the run.

        N.B. THIS ASSUMES THAT THE NEUTRAL FRACTION IS ZERO BY THE END OF THE
        INPUT RUN!
        """
        if z > 4:
            return (cosmo_factor(z) * (density_H + 3.0*density_He)).decompose()
        elif z > 5:
            return (cosmo_factor(z) * (density_H + density_He)).decompose()
        else:
            return 0

    def d_te_sim(z, xHII):
        """This is d/dz scattering depth for redshifts covered by the run.

        N.B. THIS ASSUMES THAT THE NEUTRAL FRACTION IS ZERO AT THE START OF THE
        RUN AND 1 BY THE END!
        """
        prefac = C.c * (1+z)**2 / cosmo.H(z) * thomson_cross_section
        return (prefac * (density_H*xHII + density_He*xHII)).decompose()

    sim_contrib = integrate.simps(d_te_sim(z_list, xHII), z_list)*-1
    post_sim_contrib = integrate.quad(d_te_postsim, 0, 5)
    scattering_depth = sim_contrib + post_sim_contrib

    return scattering_depth
