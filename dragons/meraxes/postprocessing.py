import logging

logging.getLogger(__name__)  # noqa

import numpy as np


def bh_bolometric_mags(gals: np.ndarray, simprops: dict, eta=0.06, quasarVoLScaling=0.0, seed=None):
    """Calculate the black hole bolometric magnitude for a set of galaxies.

    Parameters
    ----------
    gals : numpy.ndarray
        The Meraxes galaxies. Must have a compound dtype containing atleast "BlackHoleMass",
        "BlackHoleAccretedHotMass", and "BlackHoleAccretedColdMass".
    simprops : dict
        The simulation properties read with `meraxes.io.read_input_params`.
    eta : float, optional
        The accretion efficiency.
    quasarVoLScaling : float, optional
        Used for blackhole mass dependent quasar observable volume. If you don't know what this is then best to
        ignore!
    seed: int, optional
        Numpy RNG seed.

    Note
    ----
    Original code provided by Y. Qin (SNS Pisa). Modified by S. Mutch (The University of Melbourne).
    """

    #  BHM:              blackhole mass in the end (solar mass)
    #  accretedHotBHM:   accreted blackhole mass during radio mode in this time step (solar mass)
    #  accretedColdBHM:  accreted blackhole mass during quasar mode in this time step (solar mass)
    #  delta_t:          time interval of this snapshot (1e6Myr)
    #  eta:              accretion efficiency
    #  quasarVoL:        quasar view of line range (degree)
    #  quasarVoLScaling: quasar view of line range Scaling (for the purpose of BHM dependent quasarVoL)
    #  EddingtonRatio:   EddingtonRatio (taken from the simulation!)

    SOLARM2L = 14729390.502926536  # (1*units.Msun*constants.c**2./units.Myr)

    EddingtonRatio = simprops["EddingtonRatio"]
    quasarVoL = simprops["quasar_open_angle"]
    BHM = gals["BlackHoleMass"] * 1e10
    accretedHotBHM = gals["BlackHoleAccretedHotMass"] * 1e10
    accretedColdBHM = gals["BlackHoleAccretedColdMass"] * 1e10
    delta_t = gals["dt"]

    if seed:
        np.random.seed(seed=seed)

    # the stochasticity is included as following:
    # since we assume BH accretes under a constant EddingtonRatio,
    # it only glows for a fraction of this snapshot, during which the luminosity increases exponentially
    # therefore, we generage a random number, glow_time, between 0 and the snapshot time interval.
    # then the observed luminosity is the luminosity at glow_time
    # however, if glow_time is larger than the total time of accretion, accretion_time, we cannot see it
    glow_time = np.random.random(BHM.size) * delta_t

    # similarly, we assume quasar radiation is limited within a small angel, quasarVoL
    # we can only observe it if the view of line is within quasarVoL
    angle = np.random.random(BHM.size)
    solid_angle = 1.0 - np.cos(np.deg2rad(quasarVoL) / 2.0)  # normalized to 2pi
    m0 = BHM - (1.0 - eta) * accretedColdBHM  # get initial mass before accretion
    if quasarVoLScaling != 0:
        solid_angle *= (
            m0 * np.exp(EddingtonRatio * glow_time / eta / 450.0) / 1e8
        ) ** quasarVoLScaling  # if quasarVoL depends on the mass
    solid_angle[solid_angle > 1] = 1.0
    flag_undetected = angle > solid_angle  # flag_undetected=True means we cannot see this quasar

    # quasar mode
    accretion_timeq = np.log(accretedColdBHM / m0 + 1.0) * eta * 450.0 / EddingtonRatio  # get the accretion time
    QuasarLuminosity = (
        SOLARM2L * EddingtonRatio * m0 * np.exp(EddingtonRatio * glow_time / eta / 450.0) / 450.0
    )  # get the luminosity at glow_time

    # radio mode
    # do the same for radio mode, this is not significant at high-z
    m0 -= (1.0 - eta) * accretedHotBHM
    AGNLuminosity = SOLARM2L * EddingtonRatio * m0 * np.exp(EddingtonRatio * glow_time / eta / 450.0) / 450.0

    # we can also return luminosity of all black holes as well as the duty cycle factors, which can be
    # included as weights when calculating the LFs
    # !!! INCONSISTENCE of the accretion time, now since AGN luminosity is actually not used, so always use
    # quasar mode accretion time to represent the duty cycle!!!!!
    Lbol = QuasarLuminosity + AGNLuminosity * accretion_timeq / delta_t
    Lbol[flag_undetected] = 0.0

    return 4.74 - 2.5 * np.log10(Lbol)
