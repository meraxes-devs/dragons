import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astrodatapy.number_density import number_density
import seaborn as sns
from . import (
    read_snaplist,
    set_little_h,
    read_input_params,
    read_gals,
    check_for_redshift,
    read_global_xH,
    bh_bolometric_mags,
)
from .. import munge
from pathlib import Path
from typing import Union
import cycler
import click
import logging
from textwrap import dedent

logger = logging.getLogger(__name__)


class MeraxesOutput:
    """A class for dealing with Meraxes output.

    Parameters
    ----------
    fname : str or Path
        The input Meraxes master file.
    plot_dir : str or Path
        The directory where plots should be stored. (default: `./plots`)
    save : bool
        Set to `True` to save output. (default: False)
    """

    def __init__(self, fname: Union[str, Path], plot_dir: Union[str, Path] = "./plots", save: bool = False):
        self.fname = Path(fname)
        self.plot_dir = Path(plot_dir)
        self.save = save
        set_little_h(fname)
        self.snaplist, self.zlist, self.lbtimes = read_snaplist(fname)
        self.params = read_input_params(fname)

    def plot_smf(self, redshift: float, imfscaling: float = 1.0, gals: np.ndarray = None):
        """Plot the stellar mass function for a given redshift.

        Parameters
        ----------
        redshift : float
            The requested redshift to plot.
        imfscaling : float
            Scaling for IMF from Salpeter (Mstar[IMF] = Mstar[Salpeter] * imfscaling) (default: 1.0)
        gals : np.ndarray, optional
            The galaxies (already read in with correct Hubble corrections applied). If not supplied, the necessary
            galaxy properties will be read in.

        Returns
        -------
        fig : matplotlib.Figure
            The matplotlib figure
        ax : matplotlib.Axes
            The matplotlib axis
        """

        imfscaling = np.log10(imfscaling)
        snap, z = check_for_redshift(self.fname, redshift)

        logger.info(f"Plotting z={redshift:.2f} SMF")

        if gals is None:
            stellar = np.log10(read_gals(self.fname, snap, props=["StellarMass"])["StellarMass"]) + 10.0
        else:
            stellar = np.log10(gals["StellarMass"]) + 10.0

        stellar = stellar[np.isfinite(stellar)]
        smf = munge.mass_function(stellar, self.params["Volume"], 30)

        obs = number_density(feature="GSMF", z_target=z, h=self.params["Hubble_h"], quiet=True)

        fig, ax = plt.subplots(1, 1, tight_layout=True)
        alpha = 0.6
        props = cycler.cycler(marker=("o", "s", "H", "P", "*", "^", "v", "<", ">"))
        for ii, prop in zip(range(obs.n_target_observation), props):
            data = obs.target_observation["Data"][ii]
            data[:, 0] += imfscaling
            label = obs.target_observation.index[ii]
            datatype = obs.target_observation["DataType"][ii]
            data[:, 1:] = np.log10(data[:, 1:])
            if datatype == "data":
                ax.errorbar(
                    data[:, 0],
                    data[:, 1],
                    yerr=[data[:, 1] - data[:, 3], data[:, 2] - data[:, 1]],
                    label=label,
                    ls="",
                    mec="w",
                    alpha=alpha,
                    **prop,
                )
            elif datatype == "dataULimit":
                ax.errorbar(
                    data[:, 0],
                    data[:, 1],
                    yerr=-0.2 * data[:, 1],
                    uplims=True,
                    label=label,
                    mec="w",
                    alpha=alpha,
                    **prop,
                )
            else:
                ax.plot(data[:, 0], data[:, 1], label=label, lw=3, alpha=alpha)
                ax.fill_between(data[:, 0], data[:, 2], data[:, 3], alpha=0.4)

        ax.plot(smf[:, 0], np.log10(smf[:, 1]), ls="-", color="k", lw=4, label="Meraxes run")

        ax.legend(loc="lower left", fontsize="xx-small", ncol=2)
        ax.text(0.95, 0.95, f"z={z:.2f}", ha="right", va="top", transform=ax.transAxes)

        ax.set(
            xlim=(6, 13),
            ylim=(-8, 0.5),
            xlabel=r"$\log_{10}(M_*\ [{\rm M_{\odot}}])$",
            ylabel=r"$\log_{10}(\phi\ [{\rm Mpc^{-1}}])$",
        )

        if self.save:
            self.plot_dir.mkdir(exist_ok=True)
            sns.despine(ax=ax)
            fname = self.plot_dir / f"smf_z{redshift:.2f}.pdf"
            plt.savefig(fname)

        return fig, ax

    def plot_xHI(self):
        """Plot the neutral fraction evolution.

        Returns
        -------
        fig : matplotlib.Figure
            The matplotlib figure
        ax : matplotlib.Axes
            The matplotlib axis
        """

        logger.info(f"Plotting xHI evolution")

        snap_z5 = self.snaplist[np.argmin(np.abs(5.0 - self.zlist))]
        try:
            xHI = pd.DataFrame(dict(snap=np.arange(snap_z5 + 1), redshift=self.zlist[: snap_z5 + 1]))
            xHI["xHI"] = read_global_xH(self.fname, xHI.snap)
        except ValueError:
            logger.warning(f"No xHI values in Meraxes output file")
            return []

        if xHI.dropna().shape[0] == 0:
            logger.warning(f"No finite xHI values in Meraxes output file")
            return []

        start = xHI.query("xHI == xHI.max()").index[0]
        end = xHI.query("xHI == xHI.min()").index[0]
        xHI.loc[:start, "xHI"] = 1.0
        xHI.loc[end + 1 :] = 0.0

        fig, ax = plt.subplots(1, 1, tight_layout=True)
        xHI.plot(
            "redshift", "xHI", ls="-", label="Meraxes run", lw=4, color="k", ax=ax, legend=None,
        )

        ax.set(ylim=(0, 1), xlim=(15, 5), ylabel=r"$x_{\rm HI}$", xlabel="redshift")

        if self.save:
            self.plot_dir.mkdir(exist_ok=True)
            sns.despine(ax=ax)
            fname = self.plot_dir / "xHI.pdf"
            plt.savefig(fname)

        return fig, ax

    def plot_sfrf(self, redshift: float, imfscaling: float = 1.0, gals: np.ndarray = None):
        """Plot the star formation rate function.

        Parameters
        ----------
        redshift: float
            The redshift of interest
        imfscaling : float
            Scaling for IMF from Salpeter (Mstar[IMF] = Mstar[Salpeter] * imfscaling) (default: 1.0)
        gals : np.ndarray, optional
            The galaxies (already read in with correct Hubble corrections applied). If not supplied, the necessary
            galaxy properties will be read in.

        Returns
        -------
        fig : matplotlib.Figure
            The matplotlib figure
        ax : matplotlib.Axes
            The matplotlib axis
        """

        imfscaling = np.log10(imfscaling)
        snap, z = check_for_redshift(self.fname, redshift)

        logger.info(f"Plotting z={redshift:.2f} SFRF")

        if gals is None:
            sfr = read_gals(self.fname, snap, props=["Sfr"])["Sfr"]
        else:
            sfr = gals["Sfr"][:]

        sfr = np.log10(sfr[sfr > 0])
        sfrf = munge.mass_function(sfr, self.params["Volume"], 30)

        obs = number_density(feature="SFRF", z_target=redshift, h=self.params["Hubble_h"], quiet=True)

        fig, ax = plt.subplots(1, 1, tight_layout=True)
        alpha = 0.6
        props = cycler.cycler(marker=("o", "s", "H", "P", "*", "^", "v", "<", ">"))
        for ii, prop in zip(range(obs.n_target_observation), props):
            data = obs.target_observation["Data"][ii]
            data[:, 0] += imfscaling
            label = obs.target_observation.index[ii]
            datatype = obs.target_observation["DataType"][ii]
            data[:, 1:] = np.log10(data[:, 1:])
            if datatype == "data":
                ax.errorbar(
                    data[:, 0],
                    data[:, 1],
                    yerr=[data[:, 1] - data[:, 3], data[:, 2] - data[:, 1]],
                    label=label,
                    ls="",
                    mec="w",
                    alpha=alpha,
                    **prop,
                )
            elif datatype == "dataULimit":
                ax.errorbar(
                    data[:, 0],
                    data[:, 1],
                    yerr=-0.2 * data[:, 1],
                    uplims=True,
                    label=label,
                    mec="w",
                    alpha=alpha,
                    **prop,
                )
            else:
                ax.plot(data[:, 0], data[:, 1], label=label, lw=3, alpha=alpha)
                ax.fill_between(data[:, 0], data[:, 2], data[:, 3], alpha=0.4)

        ax.plot(sfrf[:, 0], np.log10(sfrf[:, 1]), ls="-", color="k", lw=4, label="Meraxes run")

        ax.legend(loc="lower left", fontsize="xx-small", ncol=2)
        ax.text(0.95, 0.95, f"z={z:.2f}", ha="right", va="top", transform=ax.transAxes)

        ax.set(
            xlim=(-2, 4),
            ylim=(-6, -1),
            xlabel=r"$\log_{10}(SFR_*\ [{\rm M_{\odot}/yr}])$",
            ylabel=r"$\log_{10}(\phi\ [{\rm Mpc^{-1}}])$",
        )

        if self.save:
            self.plot_dir.mkdir(exist_ok=True)
            sns.despine(ax=ax)
            fname = self.plot_dir / f"sfrf_z{redshift:.2f}.pdf"
            plt.savefig(fname)

        return fig, ax

    def plot_uvlf(self, redshift: float, mag_index: Union[int, None] = None, gals: np.ndarray = None):
        """Plot the UV luminosity function.

        Parameters
        ----------
        redshift: float
            The redshift of interest
        gals : np.ndarray, optional
            The galaxies (already read in with correct Hubble corrections applied). If not supplied, the necessary
            galaxy properties will be read in.

        Returns
        -------
        fig : matplotlib.Figure
            The matplotlib figure
        ax : matplotlib.Axes
            The matplotlib axis
        """

        snap, z = check_for_redshift(self.fname, redshift)

        logger.info(f"Plotting z={redshift:.2f} UVLF")

        if mag_index is None:
            mag_index = -1
            logger.warning(f"Assuming absolute UV mag to be {mag_index}")

            try:
                if gals is None:
                    mags = read_gals(self.fname, snap, props=["Mags"])["Mags"][:, mag_index]
                else:
                    mags = gals["Mags"][:, mag_index][:]
            except ValueError:
                logger.warning(f"No Mags values in Meraxes output file")
                return []

        mags = mags[mags < -10.0]
        lf = munge.mass_function(mags, self.params["Volume"], 30)

        obs = number_density(feature="GLF_UV", z_target=redshift, h=self.params["Hubble_h"], quiet=True)

        fig, ax = plt.subplots(1, 1, tight_layout=True)
        alpha = 0.6
        props = cycler.cycler(marker=("o", "s", "H", "P", "*", "^", "v", "<", ">"))
        for ii, prop in zip(range(obs.n_target_observation), props):
            data = obs.target_observation["Data"][ii]
            label = obs.target_observation.index[ii]
            datatype = obs.target_observation["DataType"][ii]
            data[:, 1:] = np.log10(data[:, 1:])
            if datatype == "data":
                ax.errorbar(
                    data[:, 0],
                    data[:, 1],
                    yerr=[data[:, 1] - data[:, 3], data[:, 2] - data[:, 1]],
                    label=label,
                    ls="",
                    mec="w",
                    alpha=alpha,
                    **prop,
                )
            elif datatype == "dataULimit":
                ax.errorbar(
                    data[:, 0],
                    data[:, 1],
                    yerr=-0.2 * data[:, 1],
                    uplims=True,
                    label=label,
                    mec="w",
                    alpha=alpha,
                    **prop,
                )
            else:
                ax.plot(data[:, 0], data[:, 1], label=label, lw=3, alpha=alpha)
                ax.fill_between(data[:, 0], data[:, 2], data[:, 3], alpha=0.4)

        ax.plot(lf[:, 0], np.log10(lf[:, 1]), ls="-", color="k", lw=4, label="Meraxes run")

        ax.legend(loc="lower left", fontsize="xx-small", ncol=2)
        ax.text(0.95, 0.95, f"z={z:.2f}", ha="right", va="top", transform=ax.transAxes)

        ax.set(
            xlim=(-10, -25), ylim=(-7, 0), xlabel=r"$M_{\rm UV}$", ylabel=r"$\log_{10}(\phi\ [{\rm Mpc^{-1}}])$",
        )

        if self.save:
            self.plot_dir.mkdir(exist_ok=True)
            sns.despine(ax=ax)
            fname = self.plot_dir / f"uvlf_z{redshift:.2f}.pdf"
            plt.savefig(fname)

        return fig, ax

    def plot_HImf(self, redshift: float, gals: np.ndarray = None):
        """Plot the HI mass function.

        Parameters
        ----------
        redshift: float
            The redshift of interest
        gals : np.ndarray, optional
            The galaxies (already read in with correct Hubble corrections applied). If not supplied, the necessary
            galaxy properties will be read in.

        Returns
        -------
        fig : matplotlib.Figure
            The matplotlib figure
        ax : matplotlib.Axes
            The matplotlib axis
        """

        snap, z = check_for_redshift(self.fname, redshift)
        logger.info(f"Plotting z={redshift:.2f} HImf")

        plot_obs = False
        if not 0.0 <= redshift <= 0.05:
            logger.warning(f"Currently only have HImf data for z=0.")
        else:
            plot_obs = True

        fig, ax = plt.subplots(1, 1, tight_layout=True)
        props = cycler.cycler(marker=("o", "s", "H", "P", "*", "^", "v", "<", ">"))()
        alpha = 0.6

        if plot_obs:
            # ALFALFA-Martin et al. 2010 (h=0.7)
            # Values provided by H. Kim.
            obs_hubble = 0.7
            _raw = dedent(
                """\
             6.3  -0.743   0.366
             6.5  -0.839   0.259
             6.7  -0.875   0.191
             6.9  -0.935   0.153
             7.1  -1.065   0.154
             7.3  -1.130   0.114
             7.5  -1.163   0.082
             7.7  -1.224   0.070
             7.9  -1.363   0.061
             8.1  -1.460   0.054
             8.3  -1.493   0.046
             8.5  -1.573   0.043
             8.7  -1.664   0.038
             8.9  -1.689   0.029
             9.1  -1.673   0.023
             9.3  -1.740   0.021
             9.5  -1.893   0.021
             9.7  -2.061   0.018
             9.9  -2.288   0.017
            10.1  -2.596   0.017
            10.3  -3.006   0.024
            10.5  -3.641   0.057
            10.7  -4.428   0.131
            10.9  -5.320   0.376
            """
            )
            data = np.fromstring(_raw, sep=" ").reshape(-1, 3)
            data[:, 0] -= 2 * np.log10(self.params["Hubble_h"] / obs_hubble)
            data[:, 1] += 3 * np.log10(self.params["Hubble_h"] / obs_hubble)

            ax.errorbar(
                data[:, 0],
                data[:, 1],
                yerr=data[:, 2],
                label="Martin et al. (2010)",
                ls="",
                mec="w",
                alpha=alpha,
                **next(props),
            )

            # HIPASS-Zwaan et al. 2005 (h=0.75)
            # Values provided by H. Kim.
            obs_hubble = 0.75
            _raw = dedent(
                """\
            7.186 -0.733 0.397 0.2039
            7.3345 -0.8838 0.3179 0.1816
            7.483 -1.1 0.301 0.1761
            7.6315 -1.056 0.1955 0.1343
            7.78 -1.207 0.1992 0.136
            7.9285 -1.35 0.1374 0.1042
            8.077 -1.315 0.08988 0.07443
            8.2255 -1.331 0.07159 0.06144
            8.374 -1.308 0.05789 0.05108
            8.5225 -1.31 0.04438 0.04027
            8.671 -1.455 0.04284 0.03899
            8.8195 -1.555 0.03725 0.03431
            8.968 -1.55 0.03187 0.02969
            9.1165 -1.69 0.03179 0.02962
            9.265 -1.735 0.02666 0.02512
            9.4135 -1.843 0.02456 0.02324
            9.562 -1.974 0.02352 0.02231
            9.7105 -2.166 0.02506 0.0237
            9.859 -2.401 0.02768 0.02602
            10.0075 -2.785 0.03275 0.03045
            10.156 -3.013 0.03628 0.03348
            10.3045 -3.417 0.05028 0.04506
            10.453 -4.044 0.07708 0.06544
            10.6015 -4.83 0.1562 0.1147
            10.75 -5.451 0.2567 0.1602
            """
            )
            data = np.fromstring(_raw, sep=" ").reshape(-1, 4)
            data[:, 0] -= 2 * np.log10(self.params["Hubble_h"] / obs_hubble)
            data[:, 1] += 3 * np.log10(self.params["Hubble_h"] / obs_hubble)

            ax.errorbar(
                data[:, 0],
                data[:, 1],
                yerr=[data[:, 2], data[:, 3]],
                label="Zwaan et al. (2005)",
                ls="",
                mec="w",
                alpha=alpha,
                **next(props),
            )

        if gals is None:
            HImass = np.log10(read_gals(self.fname, snap, props=["HIMass"])["HIMass"]) + 10.0
        else:
            HImass = np.log10(gals["HIMass"]) + 10.0
        HImass = HImass[np.isfinite(HImass)]
        mf = munge.mass_function(HImass, self.params["Volume"], 30)

        ax.plot(mf[:, 0], np.log10(mf[:, 1]), ls="-", color="k", lw=4, zorder=10, label="Meraxes run")

        ax.legend(loc="lower left", fontsize="xx-small", ncol=2)
        ax.text(0.95, 0.95, f"z={z:.2f}", ha="right", va="top", transform=ax.transAxes)

        ax.set(
            xlim=(7, 11),
            ylim=(-6, 0),
            xlabel=r"$\log_{10}(M_{\rm HI}\ [{\rm M_{\odot}}])$",
            ylabel=r"$\log_{10}(\phi\ [{\rm Mpc^{-1}}])$",
        )

        if self.save:
            self.plot_dir.mkdir(exist_ok=True)
            sns.despine(ax=ax)
            fname = self.plot_dir / f"HImf_z{redshift:.2f}.pdf"
            plt.savefig(fname)

        return fig, ax

    def plot_bolometric_qlf(self, redshift: float, gals: np.ndarray = None):
        """Plot the bolometric quasar luminosity function.

        Parameters
        ----------
        redshift: float
            The redshift of interest
        gals : np.ndarray, optional
            The galaxies (already read in with correct Hubble corrections applied). If not supplied, the necessary
            galaxy properties will be read in.

        Returns
        -------
        fig : matplotlib.Figure
            The matplotlib figure
        ax : matplotlib.Axes
            The matplotlib axis
        """
        snap, z = check_for_redshift(self.fname, redshift)

        logger.info(f"Plotting z={redshift:.2f} bolometric QLF.")
        logger.warning("This plotting routine is under construction and should not be trusted!")

        required_props = ["BlackHoleMass", "BlackHoleAccretedHotMass", "BlackHoleAccretedColdMass", "dt"]
        if gals is None:
            try:
                gals = read_gals(self.fname, snap, props=required_props)
            except ValueError:
                logger.warning(f"Unable to read required properties: {required_props}")
                return []
        else:
            if not all([prop in gals.dtype.names for prop in required_props]):
                logger.warning(f"Unable to read required properties: {required_props}")
                return []

        mags = bh_bolometric_mags(gals, self.params)
        lum = (4.74 - mags[np.isfinite(mags)]) / 2.5
        lf = munge.mass_function(lum, self.params["Volume"], 30)

        #  lf[:, 0] *= 1.0 - np.cos(np.deg2rad(self.params['quasar_open_angle']) / 2.0)  # normalized to 2pi

        obs = number_density(feature="QLF_bolometric", z_target=redshift, h=self.params["Hubble_h"], quiet=True)

        fig, ax = plt.subplots(1, 1, tight_layout=True)
        alpha = 0.6
        props = cycler.cycler(marker=("o", "s", "H", "P", "*", "^", "v", "<", ">"))
        for ii, prop in zip(range(obs.n_target_observation), props):
            data = obs.target_observation["Data"][ii]
            label = obs.target_observation.index[ii]
            datatype = obs.target_observation["DataType"][ii]
            data[:, 1:] = np.log10(data[:, 1:])
            if datatype == "data":
                ax.errorbar(
                    data[:, 0],
                    data[:, 1],
                    yerr=[data[:, 1] - data[:, 3], data[:, 2] - data[:, 1]],
                    label=label,
                    ls="",
                    mec="w",
                    alpha=alpha,
                    **prop,
                )
            elif datatype == "dataULimit":
                ax.errorbar(
                    data[:, 0],
                    data[:, 1],
                    yerr=-0.2 * data[:, 1],
                    uplims=True,
                    label=label,
                    mec="w",
                    alpha=alpha,
                    **prop,
                )
            else:
                ax.plot(data[:, 0], data[:, 1], label=label, lw=3, alpha=alpha)
                ax.fill_between(data[:, 0], data[:, 2], data[:, 3], alpha=0.4)

        ax.plot(lf[:, 0], np.log10(lf[:, 1]), ls="-", color="k", lw=4, label="Meraxes run")

        ax.legend(loc="lower left", fontsize="xx-small", ncol=2)
        ax.text(0.95, 0.95, f"z={z:.2f}", ha="right", va="top", transform=ax.transAxes)

        ax.set(
            xlim=(8, 18),
            ylim=(-14, -1),
            xlabel=r"$\log_{10}(L/{\rm L_{\odot}})$",
            ylabel=r"$\log_{10}(\phi\ [{\rm Mpc^{-1}}])$",
        )

        if self.save:
            self.plot_dir.mkdir(exist_ok=True)
            sns.despine(ax=ax)
            fname = self.plot_dir / f"bolometric_qlf_z{redshift:.2f}.pdf"
            plt.savefig(fname)

        return fig, ax

    def plot_bhmf(self, redshift: float, gals: np.ndarray = None):
        """Plot the black hole mass function for a given redshift.

        Parameters
        ----------
        redshift : float
            The requested redshift to plot.
        gals : np.ndarray, optional
            The galaxies (already read in with correct Hubble corrections applied). If not supplied, the necessary
            galaxy properties will be read in.

        Returns
        -------
        fig : matplotlib.Figure
            The matplotlib figure
        ax : matplotlib.Axes
            The matplotlib axis
        """

        snap, z = check_for_redshift(self.fname, redshift)

        logger.info(f"Plotting z={redshift:.2f} BHMF")

        if gals is None:
            bhm = read_gals(self.fname, snap, props=["BlackHoleMass"])["BlackHoleMass"]
        else:
            bhm = gals["BlackHoleMass"]

        bhm = np.log10(bhm[bhm > 0]) + 10.0
        bhmf = munge.mass_function(bhm, self.params["Volume"], 30)

        obs = number_density(feature="BHMF", z_target=z, h=self.params["Hubble_h"], quiet=True)

        fig, ax = plt.subplots(1, 1, tight_layout=True)
        alpha = 0.6
        props = cycler.cycler(marker=("o", "s", "H", "P", "*", "^", "v", "<", ">"))
        for ii, prop in zip(range(obs.n_target_observation), props):
            data = obs.target_observation["Data"][ii]
            label = obs.target_observation.index[ii]
            datatype = obs.target_observation["DataType"][ii]
            data[:, 1:] = np.log10(data[:, 1:])
            if datatype == "data":
                ax.errorbar(
                    data[:, 0],
                    data[:, 1],
                    yerr=[data[:, 1] - data[:, 3], data[:, 2] - data[:, 1]],
                    label=label,
                    ls="",
                    mec="w",
                    alpha=alpha,
                    **prop,
                )
            elif datatype == "dataULimit":
                ax.errorbar(
                    data[:, 0],
                    data[:, 1],
                    yerr=-0.2 * data[:, 1],
                    uplims=True,
                    label=label,
                    mec="w",
                    alpha=alpha,
                    **prop,
                )
            else:
                ax.plot(data[:, 0], data[:, 1], label=label, lw=3, alpha=alpha)
                ax.fill_between(data[:, 0], data[:, 2], data[:, 3], alpha=0.4)

        ax.plot(bhmf[:, 0], np.log10(bhmf[:, 1]), ls="-", color="k", lw=4, label="Meraxes run")

        ax.legend(loc="lower left", fontsize="xx-small", ncol=2)
        ax.text(0.95, 0.95, f"z={z:.2f}", ha="right", va="top", transform=ax.transAxes)

        ax.set(
            xlim=(4, 11),
            ylim=(-9, -1),
            xlabel=r"$\log_{10}(M_{\bullet} [{\rm M_{\odot}}])$",
            ylabel=r"$\log_{10}(\phi\ [{\rm Mpc^{-1}}])$",
        )

        if self.save:
            self.plot_dir.mkdir(exist_ok=True)
            sns.despine(ax=ax)
            fname = self.plot_dir / f"bhmf_z{redshift:.2f}.pdf"
            plt.savefig(fname)

        return fig, ax


def allplots(
    meraxes_fname: Union[str, Path],
    output_dir: Union[str, Path],
    uvindex: Union[int, None] = None,
    save: bool = False,
    imfscaling: float = 1.0,
):
    """Create all plots.

    Parameters
    ----------
    meraxes_fname : str or Path
        The input Meraxes master file.
    output_dir : str or Path
        The directory where plots should be stored. (default: `./plots`)
    save : bool
        Set to `True` to save output. (default: False)
    imfscaling : float
        Scaling for IMF from Salpeter (Mstar[model] = Mstar[Salpeter] * imfscaling) (default: 1.0)

    Returns
    -------
    plots: list
        A list of tuples of matplotlib (fig, ax) for each plot.
    """
    meraxes_output = MeraxesOutput(meraxes_fname, output_dir, save)

    plots = []
    for redshift in (8, 7, 6, 5, 4, 3, 2, 1, 0.5, 0):
        gals = None
        try:
            snap, _ = check_for_redshift(meraxes_fname, redshift)
            gals = read_gals(
                meraxes_fname,
                snap,
                props=[
                    "StellarMass",
                    "Sfr",
                    "HIMass",
                    "BlackHoleMass",
                    "BlackHoleAccretedHotMass",
                    "BlackHoleAccretedColdMass",
                    "dt",
                ],
            )
        except KeyError:
            continue

        plots.append(meraxes_output.plot_smf(redshift, imfscaling=imfscaling, gals=gals))
        plots.append(meraxes_output.plot_sfrf(redshift, imfscaling=imfscaling, gals=gals))
        plots.append(meraxes_output.plot_bhmf(redshift, gals=gals))
        plots.append(meraxes_output.plot_bolometric_qlf(redshift, gals=gals))

        if redshift == 0:
            plots.append(meraxes_output.plot_HImf(redshift, gals=gals))

        if redshift >= 4:
            # we don't pass gals here as the presence of mags is not guaranteed
            plots.append(meraxes_output.plot_uvlf(redshift, uvindex))

    plots.append(meraxes_output.plot_xHI())

    return plots


@click.command()
@click.argument("meraxes_fname", type=click.Path(exists=True))
@click.option("--output_dir", "-o", type=click.Path(), default="plots")
@click.option("--uvindex", type=click.INT)
@click.option(
    "--imfscaling",
    type=click.FLOAT,
    help="Scaling for IMF from Salpeter (Mstar[IMF] = Mstar[Salpeter] * imfscaling).",
    default=1.0,
)
def main(meraxes_fname, output_dir="plots", uvindex=None, imfscaling=1.0):
    import warnings
    import os

    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), handlers=logging.getLogger("dragons").handlers)
    logging.getLogger("dragons.meraxes.io").setLevel("ERROR")

    sns.set(
        "talk", "ticks", font_scale=1.2, rc={"lines.linewidth": 3, "figure.figsize": (12, 6)},
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module=__name__)
        allplots(meraxes_fname, output_dir, uvindex, True, imfscaling=imfscaling)


if __name__ == "__main__":
    main()
