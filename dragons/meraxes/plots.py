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
)
from .. import munge
from pathlib import Path
from typing import Union
import cycler
import click
import logging

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

    def plot_smf(self, redshift: float, imfscaling: float = 1.0):
        """Plot the stellar mass function for a given redshift.

        Parameters
        ----------
        redshift : float
            The requested redshift to plot.
        imfscaling : float
            Scaling for IMF from Salpeter (Mstar[IMF] = Mstar[Salpeter] * imfscaling) (default: 1.0)

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

        stellar = np.log10(read_gals(self.fname, snap, props=["StellarMass"])["StellarMass"]) + 10.0
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

    def plot_sfrf(self, redshift: float, imfscaling: float = 1.0):
        """Plot the star formation rate function.

        Parameters
        ----------
        redshift: float
            The redshift of interest
        imfscaling : float
            Scaling for IMF from Salpeter (Mstar[IMF] = Mstar[Salpeter] * imfscaling) (default: 1.0)

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

        sfr = read_gals(self.fname, snap, props=["Sfr"])["Sfr"]
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

    def plot_uvlf(self, redshift: float, mag_index: Union[int, None] = None):
        """Plot the UV luminosity function.

        Parameters
        ----------
        redshift: float
            The redshift of interest

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
            mags = read_gals(self.fname, snap, props=["Mags"])["Mags"][:, mag_index]
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


def allplots(
    meraxes_fname: Union[str, Path], output_dir: Union[str, Path], uvindex: Union[int, None] = None, save: bool = False,
    imfscaling: float = 1.0
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
        Scaling for IMF from Salpeter (Mstar[IMF] = Mstar[Salpeter] * imfscaling) (default: 1.0)

    Returns
    -------
    plots: list
        A list of tuples of matplotlib (fig, ax) for each plot.
    """
    meraxes_output = MeraxesOutput(meraxes_fname, output_dir, save)

    plots = []
    for redshift in (8, 7, 6, 5, 4, 3, 2, 1, 0.5, 0):
        try:
            plots.append(meraxes_output.plot_smf(redshift, imfscaling=imfscaling))
            plots.append(meraxes_output.plot_sfrf(redshift, imfscaling=imfscaling))
        except KeyError:
            pass

    for redshift in (8, 7, 6, 5, 4):
        try:
            plots.append(meraxes_output.plot_uvlf(redshift, uvindex))
        except KeyError:
            pass

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
    default=1.0
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
