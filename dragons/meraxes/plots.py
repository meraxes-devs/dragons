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
    def __init__(
        self, fname: Union[str, Path], plot_dir: Union[str, Path] = './plots', save: bool = False
    ):
        self.fname = Path(fname)
        self.plot_dir = Path(plot_dir)
        self.save = save
        set_little_h(fname)
        self.snaplist, self.zlist, self.lbtimes = read_snaplist(fname)
        self.params = read_input_params(fname)

    def plot_smf(self, redshift: float):
        """Plot the stellar mass function for a given redshift.

        Parameters
        ----------
        redshift : float
            The requested redshift to plot.

        Returns
        -------
        fig : matplotlib.Figure
            The matplotlib figure
        ax : matplotlib.Axes
            The matplotlib axis
        """
        snap, z = check_for_redshift(self.fname, redshift)
        stellar = (
            np.log10(
                read_gals(self.fname, snap, props=["StellarMass"], pandas=True)[
                    "StellarMass"
                ]
            )
            + 10.0
        )
        stellar = stellar[np.isfinite(stellar)]
        smf = munge.mass_function(stellar, self.params["Volume"], 30)

        obs = number_density(
            feature="GSMF", z_target=z, quiet=1, h=self.params["Hubble_h"]
        )

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

        ax.plot(
            smf[:, 0], np.log10(smf[:, 1]), ls="-", color="k", lw=4, label="Meraxes run"
        )

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
        snap_z5 = self.snaplist[np.argmin(np.abs(5.0 - self.zlist))]
        xHI = pd.DataFrame(
            dict(
                snap=np.arange(snap_z5 + 1), redshift=self.zlist[: snap_z5 + 1], xHI=0.0
            )
        )
        xHI.xHI = read_global_xH(self.fname, xHI.snap, quiet=True)

        start = xHI.query("xHI == xHI.max()").index[0]
        end = xHI.query("xHI == xHI.min()").index[0]
        xHI.loc[:start, "xHI"] = 1.0
        xHI.loc[end + 1 :] = 0.0

        fig, ax = plt.subplots(1, 1, tight_layout=True)
        fig: plt.Figure
        ax: plt.Axes
        xHI.plot(
            "redshift",
            "xHI",
            ls="-",
            label="Meraxes run",
            lw=4,
            color="k",
            ax=ax,
            legend=None,
        )

        ax.set(ylim=(0, 1), xlim=(15, 5), ylabel=r"$x_{\rm HI}$", xlabel="redshift")

        if self.save:
            self.plot_dir.mkdir(exist_ok=True)
            sns.despine(ax=ax)
            fname = self.plot_dir / "xHI.pdf"
            plt.savefig(fname)

        return fig, ax


def allplots(
    meraxes_fname: Union[str, Path], output_dir: Union[str, Path], save: bool = False
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

    Returns
    -------
    plots: list
        A list of tuples of matplotlib (fig, ax) for each plot.
    """
    meraxes_output = MeraxesOutput(meraxes_fname, output_dir, save)

    plots = []
    for redshift in (8, 7, 6, 5, 4, 3, 2, 1, 0.5, 0):
        try:
            plots.append(meraxes_output.plot_smf(redshift))
        except KeyError:
            pass

    plots.append(meraxes_output.plot_xHI())

    return plots


@click.command()
@click.argument("meraxes_fname", type=click.Path(exists=True))
@click.option("--output_dir", "-o", type=click.Path(), default="plots")
def main(meraxes_fname, output_dir="plots"):
    sns.set(
        "talk",
        "ticks",
        font_scale=1.2,
        rc={"lines.linewidth": 3, "figure.figsize": (12, 6)},
    )
    allplots(meraxes_fname, output_dir, True)


if __name__ == "__main__":
    main()
