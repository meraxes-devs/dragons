#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Routines for reionisation related calculations."""

import warnings
import numpy as np
from scipy import integrate
from tqdm import tqdm
from astropy import cosmology
from astropy import units as U
from astropy import constants as C
from .io import read_input_params, read_snaplist, read_global_xH, read_grid


def electron_optical_depth(fname, volume_weighted=False):
    """Calculate the electron Thomson scattering optical depth from a Meraxes +
    21cmFAST run.  Note that this implementation assumes that the simulation
    volume is fully ionised before the final snapshot stored in the input file.

    Parameters
    ----------
    fname : str
        Full path to input hdf5 master file

    volume_weighted : bool
        This option is just for testing purposes as it can take a long time
        to mass weight the neutral fraction depending on the grid size.
        The optical depth obtained is often very similar to the correctly
        mass weighted value, however, this should not be used to final
        results.

    Returns
    -------
    z_list : ndarray
        Redshifts of each snapshot read in the input simulation.

    scattering_depth : ndarray
        Thomson scattering depth integrated between z=0 and each snapshot
        of the input simulation.
    """

    # read in the model run parameters and set up the cosmology
    run_params = read_input_params(fname)
    cosmo = cosmology.FlatLambdaCDM(H0=run_params['Hubble_h']*100.,
                                    Om0=run_params['OmegaM'],
                                    Ob0=run_params['OmegaM']
                                    * run_params['BaryonFrac'])

    # define the necessary constants
    thomson_cross_section = 6.652e-25 * U.cm * U.cm
    density_H = 1.88e-7 * cosmo.Ob0 * cosmo.h**2 / 0.022 * U.cm**-3
    # density_He = 0.19e-7 * cosmo.Ob0 * cosmo.h**2 / 0.022 * U.cm**-3
    # Not what is in Whythe et al.!
    density_He = 0.148e-7 * cosmo.Ob0 * cosmo.h**2 / 0.022 * U.cm**-3

    cosmo_factor = lambda z: C.c * (1+z)**2 \
        / cosmo.H(z) * thomson_cross_section

    # read in the model run data
    snaps, z_list, _ = read_snaplist(fname)
    if not volume_weighted:        
        xHII = 1.0 - read_global_xH(fname, snaps, weight='mass', quiet=True)
    else:
        xHII = 1.0 - read_global_xH(fname, snaps, weight='volume', quiet=True)
        
    first_valid, last_valid = np.where(~np.isnan(xHII))[0][[0, -1]]
    xHII[:first_valid] = 0.0
    xHII[last_valid+1:] = 1.0

    # reorder the run data from low z to high z for ease of integration
    xHII = xHII[::-1]
    z_list = z_list[::-1]

    def d_te_postsim(z):
        """This is d/dz scattering depth for redshifts greater than the final
        redshift of the run.

        N.B. THIS ASSUMES THAT THE NEUTRAL FRACTION IS ZERO BY THE END OF THE
        INPUT RUN!
        """
        if z <= 4:
            return (cosmo_factor(z) * (density_H + 2.0*density_He)).decompose()
        else:
            return (cosmo_factor(z) * (density_H + density_He)).decompose()

    def d_te_sim(z, xHII):
        """This is d/dz scattering depth for redshifts covered by the run.
        """
        prefac = cosmo_factor(z)
        return (prefac * (density_H*xHII + (1 + (z<=4))*density_He*xHII)).decompose()
        
    post_sim_contrib = integrate.quad(d_te_postsim, 0, z_list[0])[0]

    sim_contrib = np.zeros(z_list.size)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim_contrib = np.array([integrate.simps(d_te_sim(z_list[:ii+1],
                                                         xHII[:ii+1]),
                                                z_list[:ii+1])
                                for ii in range(z_list.size)])

    scattering_depth = sim_contrib + post_sim_contrib

    return z_list, scattering_depth
