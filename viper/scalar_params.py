"""
Model input parameters with validation.

Provides the PWNScalarParams dataclass that encapsulates all physical
parameters needed to run the Gelfand et al. (2009) PWN evolution model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from . import constants as C
from .pulsar import true_age_yr, e0_37_from_timing

logger = logging.getLogger(__name__)


def check_physical_inputs(*, esn, mej, n_ism, braking_index, tau_sd,
                     period, p_dot, eta_b, eta_g, e_min, e_max,
                     e_break, p1, p2, f_max, vel_psr,
                     kT_max=0.0, distance=1000.0,
                     n_ic=0, ic_temps=(), ic_norms=()) -> list[str]:
    """Shared physics-range validation for BOTH scalar APIs.

    Single source of truth for ``PWNScalarParams.validate`` and
    ``PWNModel.validate``. Returns an ordered list of problem strings
    (empty = valid). Suspicious-but-legal values (negative
    braking_index/p1/p2) log warnings.

    Checks: kT_max > 0 when f_max > 0; distance positive and finite;
    ic_temps positive and finite; ic_norms >= 0 and finite (norm 0 =
    valid no-op).
    """
    problems = []
    if esn <= 0:
        problems.append(f"esn must be > 0, got {esn}")
    if mej <= 0:
        problems.append(f"mej must be > 0, got {mej}")
    if n_ism <= 0:
        problems.append(f"n_ism must be > 0, got {n_ism}")
    if tau_sd <= 0:
        problems.append(f"tau_sd must be > 0, got {tau_sd}")
    if period <= 0:
        problems.append(f"period must be > 0, got {period}")
    if p_dot <= 0:
        problems.append(f"p_dot must be > 0, got {p_dot}")
    if eta_b < 0 or eta_b > 1:
        problems.append(f"eta_b must be in [0, 1], got {eta_b}")
    if eta_g < 0 or eta_g >= 1:
        problems.append(f"eta_g must be in [0, 1), got {eta_g}")
    if e_min < 0:
        problems.append(f"e_min must be >= 0, got {e_min}")
    if e_max < e_min:
        problems.append(f"e_max must be >= e_min, got {e_max}")
    if e_break != 0:
        if e_break < 0:
            problems.append(f"e_break must be >= 0, got {e_break}")
        if e_break >= e_max:
            problems.append(
                f"e_break must be < e_max, got {e_break} >= {e_max}"
            )
    if f_max < 0 or f_max > 1:
        problems.append(f"f_max must be in [0, 1], got {f_max}")
    if kT_max < 0 or not np.isfinite(kT_max):
        problems.append(f"kT_max must be >= 0 and finite, got {kT_max}")
    elif f_max > 0 and kT_max == 0:
        problems.append(
            f"kT_max must be > 0 when f_max > 0 (Maxwellian component "
            f"temperature), got kT_max=0 with f_max={f_max}"
        )
    if vel_psr < 0:
        problems.append(f"vel_psr must be >= 0, got {vel_psr}")
    if distance <= 0 or not np.isfinite(distance):
        problems.append(
            f"distance must be positive and finite, got {distance}"
        )
    if n_ic > 0:
        if len(ic_temps) != n_ic:
            problems.append(
                f"ic_temps length ({len(ic_temps)}) != n_ic ({n_ic})"
            )
        if len(ic_norms) != n_ic:
            problems.append(
                f"ic_norms length ({len(ic_norms)}) != n_ic ({n_ic})"
            )
        for i, t in enumerate(ic_temps):
            if t <= 0 or not np.isfinite(t):
                problems.append(
                    f"ic_temps[{i}] must be positive and finite, got {t}"
                )
        for i, nrm in enumerate(ic_norms):
            if nrm < 0 or not np.isfinite(nrm):
                problems.append(
                    f"ic_norms[{i}] must be >= 0 and finite, got {nrm}"
                )

    if braking_index == 1:
        problems.append(
            "braking_index must not be 1 (division by zero in spindown_index)"
        )
    if braking_index < 0:
        logger.warning("braking_index = %f < 0. Are you sure?", braking_index)
    if p1 < 0:
        logger.warning("p1 = %f < 0. Are you sure?", p1)
    if p2 < 0:
        logger.warning("p2 = %f < 0. Are you sure?", p2)
    return problems


@dataclass
class PWNScalarParams:
    """Input parameters for the PWN evolution model.

    Parameters
    ----------
    esn : float
        Initial kinetic energy of supernova ejecta [10^51 erg].
    mej : float
        Mass of supernova ejecta [solar masses].
    n_ism : float
        Number density of surrounding ISM [cm^-3].
    braking_index : float
        Pulsar braking index (dimensionless).
    tau_sd : float
        Pulsar spin-down timescale [years].
    eta_b : float
        Magnetisation of the pulsar wind (fraction of wind energy in B field).
    e_min : float
        Minimum particle energy in pulsar wind [GeV].
    e_max : float
        Maximum particle energy in pulsar wind [GeV].
    e_break : float
        Break energy in pulsar wind [GeV]. Set to 0 for a single power law.
    p1 : float
        Injection spectral index below the break (dN/dE ~ E^{-p1}).
    p2 : float
        Injection spectral index above the break (dN/dE ~ E^{-p2}).
    period : float
        Pulsar spin period [s].
    p_dot : float
        Period derivative [s/s].
    distance : float
        Distance to PWN [pc]. Used only for flux conversions, not dynamics.
    eta_g : float
        Fraction of spin-down luminosity lost to radiation before wind.
    vel_psr : float
        Space velocity of pulsar [cm/s]. Default 0.
    f_max : float
        Fraction of wind particle energy in Maxwellian component. Default 0.
    kT_max : float
        Temperature of Maxwellian component [GeV]. Default 0.
    n_ic : int
        Number of additional (non-CMB) background photon fields for IC.
    ic_temps : list of float
        Temperatures of additional IC photon fields [K].
    ic_norms : list of float
        Normalizations of additional IC photon fields.
    n_elec_bins : int
        Number of electron energy bins. Default 500.
    n_phot_bins : int
        Number of output photon frequency bins. Default 500.
    n_ic_target : int
        Number of IC target photon frequency bins. Default 200.
    """

    # Supernova / ISM
    esn: float  # 10^51 erg
    mej: float  # solar masses
    n_ism: float  # cm^-3

    # Pulsar
    braking_index: float
    tau_sd: float  # years
    period: float  # seconds
    p_dot: float  # s/s

    # Pulsar wind particle spectrum
    eta_b: float
    e_min: float  # GeV
    e_max: float  # GeV
    e_break: float  # GeV (0 for single PL)
    p1: float
    p2: float

    # Distance (for flux conversions)
    distance: float = 1000.0  # pc

    # Optional physics
    eta_g: float = 0.0
    vel_psr: float = 0.0  # cm/s
    f_max: float = 0.0
    kT_max: float = 0.0  # GeV

    # IC background fields beyond CMB
    n_ic: int = 0
    ic_temps: list[float] = field(default_factory=list)
    ic_norms: list[float] = field(default_factory=list)

    # Grid resolution
    n_elec_bins: int = C.N_ELEC_BINS
    n_phot_bins: int = C.N_PHOT_FREQ
    n_ic_target: int = C.N_IC_TARGET
    phot_freq_min: float = C.PHOT_FREQ_MIN
    phot_freq_max: float = C.PHOT_FREQ_MAX
    target_freq_min: float = C.TARGET_FREQ_MIN
    target_freq_max: float = C.TARGET_FREQ_MAX

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate parameter values, raising ValueError for invalid inputs.

        Delegates to :func:`check_physical_inputs` and raises on the
        first problem.
        """
        problems = check_physical_inputs(
            esn=self.esn, mej=self.mej, n_ism=self.n_ism,
            braking_index=self.braking_index, tau_sd=self.tau_sd,
            period=self.period, p_dot=self.p_dot,
            eta_b=self.eta_b, eta_g=self.eta_g,
            e_min=self.e_min, e_max=self.e_max, e_break=self.e_break,
            p1=self.p1, p2=self.p2,
            f_max=self.f_max, vel_psr=self.vel_psr,
            kT_max=self.kT_max, distance=self.distance,
            n_ic=self.n_ic, ic_temps=self.ic_temps,
            ic_norms=self.ic_norms,
        )
        if problems:
            raise ValueError(problems[0])

    # -----------------------------------------------------------------------
    # Derived quantities
    # -----------------------------------------------------------------------
    @property
    def true_age(self) -> float:
        """True age of the neutron star [years]."""
        return true_age_yr(self.period, self.p_dot,
                           self.braking_index, self.tau_sd)

    @property
    def e0_37(self) -> float:
        """Initial spin-down luminosity [10^37 erg/s]."""
        return e0_37_from_timing(self.period, self.p_dot,
                                 self.braking_index, self.tau_sd)

    @property
    def spindown_index(self) -> float:
        """Spin-down index n = (p+1)/(p-1) where p is the braking index."""
        return (self.braking_index + 1) / (self.braking_index - 1)

    @property
    def esd_40(self) -> float:
        """Total rotational energy in units of 10^40 erg."""
        return self.e0_37 * 1.0e-3 * self.tau_sd * C.yr

    @classmethod
    def from_theta(
        cls,
        theta: Sequence[float],
        period: float,
        p_dot: float,
        distance: Optional[float] = None,
        ic_fields: Optional[Sequence[float]] = None,
    ) -> PWNScalarParams:
        """Create from the 12-element theta vector used by the fitting layer.

        Parameters
        ----------
        theta : sequence of 12 floats
            [esn, mej, n_ism, distance, braking_index, tau_sd,
             eta_b, e_min, e_max, e_break, p1, p2]
        period : float
            Pulsar spin period [s].
        p_dot : float
            Period derivative [s/s].
        distance : float, optional
            Override distance from theta[3].
        ic_fields : sequence of floats, optional
            Additional IC photon fields: [n_fields, T1, norm1, T2, norm2, ...].
        """
        if len(theta) < 12:
            raise ValueError(f"theta must have at least 12 elements, got {len(theta)}")
        dist = distance if distance is not None else theta[3]

        n_ic = 0
        ic_temps = []
        ic_norms = []
        if ic_fields is not None and len(ic_fields) >= 3:
            n_ic = int(ic_fields[0])
            for i in range(n_ic):
                ic_temps.append(ic_fields[1 + 2 * i])
                ic_norms.append(ic_fields[2 + 2 * i])

        return cls(
            esn=theta[0],
            mej=theta[1],
            n_ism=theta[2],
            distance=dist,
            braking_index=theta[4],
            tau_sd=theta[5],
            eta_b=theta[6],
            e_min=theta[7],
            e_max=theta[8],
            e_break=theta[9],
            p1=theta[10],
            p2=theta[11],
            period=period,
            p_dot=p_dot,
            n_ic=n_ic,
            ic_temps=ic_temps,
            ic_norms=ic_norms,
        )
