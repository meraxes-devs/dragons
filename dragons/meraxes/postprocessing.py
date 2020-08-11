import logging
import numpy as np
from astropy import units as U, constants as C

logging.getLogger(__name__)  # noqa


def bh_bolometric_mags(
    gals: np.ndarray, simprops: dict, eta=0.06, quasarVoLScaling=0.0, seed=None, consider_opening_angle=False,
):
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
    consider_opening_angle : bool, optional (default: False)
        Should we consider a random orientation for each QSO and decide if we can observe it based on the opening angle?
        Note that `consider_opening_angle = True` by default if `quasarVoLScaling > 0.0`. See also the note below.

    Note
    ----
    - Original code provided by Y. Qin (SNS Pisa). Modified by S. Mutch (The University of Melbourne).

    - If `consider_opening_angle = False` and `quasarVoLScaling == 0.0` then the opening angle can still be taken in to
    account when generating a luminoisty function by scaling the ɸ values by
    (1.0 - np.cos(np.deg2rad(simprops['quasar_open_angle']) / 2.0)).
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

    m0 = BHM - (1.0 - eta) * accretedColdBHM  # get initial mass before accretion

    # similarly, we assume quasar radiation is limited within a small angel, quasarVoL
    # we can only observe it if the view of line is within quasarVoL
    flag_undetected = np.zeros(BHM.size, bool)
    if consider_opening_angle or quasarVoLScaling > 0.0:
        angle = np.random.random(BHM.size)
        solid_angle = 1.0 - np.cos(np.deg2rad(quasarVoL) / 2.0)  # normalized to 2pi
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
    Lbol = (QuasarLuminosity + AGNLuminosity) * accretion_timeq / delta_t
    Lbol[flag_undetected] = 0.0
    with np.errstate(divide="ignore"):
        Mbol = 4.74 - 2.5 * np.log10(Lbol)

    return Mbol


def bh_radio_lum(gals: np.ndarray):
    """
    TODO: Document this function.
    TODO: Allow SPIN to be claculated from efficiencies.
    """
    ARADIO = 8.0e-5
    AQSO = 5.0e-2
    SPIN = 0.083  # NB This corresponds to ε=0.06
    FREQ = 1.4e9  # Hz

    m_bh = gals["BlackHoleMass"] * 10.0  # 1e9 Msun
    mdot_acc_hot = gals["BlackHoleAccretedHotMass"] * 10 / gals["dt"]  # 1e9 Msun / Myr
    mdot_edd = (m_bh * 1e9 * U.Msun * 4.0 * np.pi * U.G * C.m_p * C.c / C.sigma_T).to("1e9 Msun Myr-1").value
    mdot_ratio = mdot_acc_hot / mdot_edd  # unitless

    Ljet_radio = 2.0e45 * m_bh * (mdot_ratio / 0.01) * SPIN ** 2  # erg s-1
    νL_radio = ARADIO * (m_bh * (mdot_ratio / 0.01)) ** 0.42 * Ljet_radio  # W
    νL_qso = AQSO * m_bh ** 1.42 * 2.5e43 * SPIN ** 2  # W

    L_tot = (νL_radio + νL_qso) / FREQ  # W Hz-1
    with np.errstate(divide="ignore"):
        L_tot = np.log10()

    return L_tot
