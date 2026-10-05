"""
Top-level interface to the PWN evolution model.

PWNModel holds physical parameters as Parameter objects (units,
frozen flag, log sampling) for forward evaluation and MCMC fitting.
"""

from __future__ import annotations

import copy
import logging
from typing import Optional, Sequence

import numpy as np

from .parameter import Parameter, ParameterSet
from .evolution import evolve_pwn, PWNModelResult
from .scalar_params import check_physical_inputs
from .pulsar import true_age_yr, e0_37_from_timing
from . import constants as C

logger = logging.getLogger(__name__)


# Internal unit definitions

def _get_units():
    """Lazy-load astropy units to avoid import cost when not needed."""
    import astropy.units as u
    return u


# Parameter definitions table
# Each entry: (name, default, frozen, log, unit_str, scale, description)

_PARAM_DEFS = [
    # -- Supernova / ISM --
    ('esn',     1.0,   True,  True,  'erg',    1e-51, 'SN kinetic energy [internal: 10^51 erg]'),
    ('mej',     6.0,   True,  True,  'Msun',   1.0,   'Ejecta mass [solar masses]'),
    ('n_ism',   0.4,   True,  True,  '1/cm3',  1.0,   'ISM number density [cm^-3]'),
    # -- Pulsar --
    ('period',  None,  True,  False, 's',      1.0,   'Spin period [s]'),
    ('p_dot',   None,  True,  False, None,     1.0,   'Period derivative [s/s, dimensionless]'),
    ('braking_index', 2.5, True, False, None,  1.0,   'Braking index'),
    ('tau_sd',  3000., True,  True,  'yr',     1.0,   'Spin-down timescale [yr]'),
    # -- Pulsar wind --
    ('eta_b',   0.01,  True,  True,  None,     1.0,   'Wind magnetisation fraction'),
    ('e_min',   3.0,   True,  True,  'GeV',    1.0,   'Min particle energy [GeV]'),
    ('e_max',   5e5,   True,  True,  'GeV',    1.0,   'Max particle energy [GeV]'),
    ('e_break', 50.0,  True,  True,  'GeV',    1.0,   'Break energy [GeV], 0 = single PL'),
    ('p1',      0.8,   True,  False, None,     1.0,   'Injection index below break'),
    ('p2',      2.5,   True,  False, None,     1.0,   'Injection index above break'),
    # -- Distance --
    ('distance', 1000., True, False, 'pc',     1.0,   'Distance [pc]'),
    # -- Optional physics --
    ('eta_g',   0.0,   True,  False, None,     1.0,   'Radiative loss fraction'),
    ('vel_psr', 0.0,   True,  False, 'cm/s',   1.0,   'Pulsar space velocity [cm/s]'),
    ('f_max',   0.0,   True,  False, None,     1.0,   'Maxwellian fraction'),
    ('kT_max',  0.0,   True,  False, 'GeV',    1.0,   'Maxwellian temperature [GeV]'),
    # -- Particle escape --
    ('esc_norm',  1.0,  True,  True,  None,     1.0,   'Escape normalization (dimensionless multiplier)'),
    ('esc_index', 0.0,  True,  False, None,     1.0,   'Escape energy index (delta/s)'),
]


def _build_default_parameters() -> ParameterSet:
    """Create the default parameter set from definitions."""
    u = None  # lazy
    params = []
    for name, default, frozen, log, unit_str, scale, desc in _PARAM_DEFS:
        unit = None
        if unit_str is not None:
            if u is None:
                u = _get_units()
            unit = u.Unit(unit_str)
        val = default if default is not None else np.nan
        params.append(Parameter(
            name=name, default=val, frozen=frozen, log=log,
            unit=unit, scale=scale, description=desc,
        ))
    return ParameterSet(params)


# PWNModel

class PWNModel:
    """Gelfand et al. (2009) PWN evolution model.

    Physical parameters are attributes returning Parameter objects;
    set with astropy Quantities via ``.value``.

    Parameters
    ----------
    None.  Use ``set_pulsar()``, then set values and bounds before
    ``run()`` or MCMC fitting.
    """

    def __init__(self) -> None:
        self._parameters = _build_default_parameters()
        self._ic_count = 0
        self._result: Optional[PWNModelResult] = None
        # Grid settings (not fitted)
        self.n_elec_bins: int = C.N_ELEC_BINS
        self.n_phot_bins: int = C.N_PHOT_FREQ
        self.n_ic_target: int = C.N_IC_TARGET
        self.phot_freq_min: float = C.PHOT_FREQ_MIN
        self.phot_freq_max: float = C.PHOT_FREQ_MAX
        self.target_freq_min: float = C.TARGET_FREQ_MIN
        self.target_freq_max: float = C.TARGET_FREQ_MAX
        # CMB IC lookup; falls through to live IC when incompatible.
        self.use_cmb_lookup: bool = False
        # Fit spec captures via getattr.
        self.escape_model: str = 'off'
        # Non-default injection forms reuse e_break/p1/p2.
        self.injection_model: str = 'broken_powerlaw'
        self.ssc_model: str = 'off'

    # Parameter access

    def __getattr__(self, name: str):
        # _parameters must exist or lookup recurses.
        if name.startswith('_'):
            raise AttributeError(name)
        try:
            return self._parameters[name]
        except KeyError:
            raise AttributeError(
                f"'{type(self).__name__}' has no parameter '{name}'"
            ) from None

    def __setattr__(self, name: str, value) -> None:
        # Block Parameter shadowing.
        params = self.__dict__.get('_parameters')
        if params is not None and name in params:
            raise AttributeError(
                f"'{name}' is a model Parameter and cannot be replaced "
                f"by assignment. Did you mean "
                f"model.{name}.value = ...?"
            )
        super().__setattr__(name, value)

    @property
    def parameters(self) -> ParameterSet:
        """All model parameters."""
        return self._parameters

    # Pulsar setup

    def set_pulsar(
        self,
        name: Optional[str] = None,
        *,
        period=None,
        p_dot=None,
        distance=None,
    ) -> None:
        """Set pulsar timing parameters.

        Parameters
        ----------
        name : str, optional
            ATNF catalogue name (J- or B-name); loads period, p_dot,
            distance.  Manual keywords override.
        period : Quantity, optional
            Spin period (e.g. ``0.316 * u.s``).
        p_dot : float, optional
            Period derivative [s/s] (dimensionless).
        distance : Quantity, optional
            Distance (e.g. ``1.4 * u.kpc``).
        """
        if name is not None:
            from .catalog import PulsarCatalog
            cat = PulsarCatalog()
            entry = cat[name]
            # Internal units; bypass Quantity validation.
            if entry.period is not None and period is None:
                self._parameters['period']._set_internal(entry.period)
            if entry.p_dot is not None and p_dot is None:
                self._parameters['p_dot']._set_internal(entry.p_dot)
            if entry.dist is not None and distance is None:
                # kpc -> pc
                self._parameters['distance']._set_internal(
                    entry.dist * 1000.0
                )
            elif distance is None:
                logger.warning(
                    "%s has no distance in the catalogue; distance stays "
                    "at %.0f pc", entry.jname,
                    self._parameters['distance'].value)
            logger.info(
                "Loaded %s: P=%.6f s, Pdot=%.3e s/s, d=%.0f pc",
                entry.jname,
                self._parameters['period'].value,
                self._parameters['p_dot'].value,
                self._parameters['distance'].value,
            )

        # Manual overrides
        if period is not None:
            self._parameters['period'].value = period
        if p_dot is not None:
            self._parameters['p_dot'].value = p_dot
        if distance is not None:
            self._parameters['distance'].value = distance

    # IC photon fields

    def add_ic_field(
        self,
        temp,
        norm,
        frozen: bool = True,
        temp_log: bool = True,
        norm_log: bool = True,
    ) -> int:
        """Add an additional IC background photon field.

        Parameters
        ----------
        temp : Quantity or float
            Temperature; bare float means Kelvin.
        norm : Quantity or float
            Total energy density of the field; bare float means eV/cm^3.
        frozen : bool
            Whether both temp and norm are frozen.
        temp_log, norm_log : bool
            Whether to sample in log space.

        Returns
        -------
        int
            The field index (1-based).

        Raises
        ------
        ValueError, astropy.units.UnitConversionError
            If ``temp`` or ``norm`` cannot be converted; the model is
            left unchanged.
        """
        u = _get_units()

        # Convert both before mutating state.
        if hasattr(temp, 'to') and hasattr(temp, 'unit'):
            temp_val = temp.to(u.K).value
        else:
            temp_val = float(temp)

        norm_unit = u.eV / u.cm**3
        if hasattr(norm, 'to') and hasattr(norm, 'unit'):
            norm_val = norm.to(norm_unit).value
        else:
            norm_val = float(norm)

        self._ic_count += 1
        idx = self._ic_count

        self._parameters.add(Parameter(
            name=f'ic_temp_{idx}',
            default=temp_val,
            frozen=frozen,
            log=temp_log,
            unit=u.K,
            description=f'IC field {idx} temperature [K]',
        ))

        self._parameters.add(Parameter(
            name=f'ic_norm_{idx}',
            default=norm_val,
            frozen=frozen,
            log=norm_log,
            unit=norm_unit,
            description=f'IC field {idx} energy density [eV/cm^3]',
        ))

        return idx

    @property
    def n_ic(self) -> int:
        """Number of additional IC photon fields."""
        return self._ic_count

    @property
    def ic_temps(self) -> list[float]:
        """IC field temperatures [K] in order."""
        return [
            self._parameters[f'ic_temp_{i+1}'].value
            for i in range(self._ic_count)
        ]

    @property
    def ic_norms(self) -> list[float]:
        """IC field normalizations in order."""
        return [
            self._parameters[f'ic_norm_{i+1}'].value
            for i in range(self._ic_count)
        ]

    # Derived quantities

    @property
    def true_age(self) -> float:
        """True age of the neutron star [years]."""
        return true_age_yr(
            self._parameters['period'].value,
            self._parameters['p_dot'].value,
            self._parameters['braking_index'].value,
            self._parameters['tau_sd'].value,
        )

    @property
    def e0_37(self) -> float:
        """Initial spin-down luminosity [10^37 erg/s]."""
        return e0_37_from_timing(
            self._parameters['period'].value,
            self._parameters['p_dot'].value,
            self._parameters['braking_index'].value,
            self._parameters['tau_sd'].value,
        )

    @property
    def esd_40(self) -> float:
        """Total rotational energy [10^40 erg]."""
        return self.e0_37 * 1e-3 * self._parameters['tau_sd'].value * C.yr

    # Model evaluation

    def validate(self) -> None:
        """Check that all required parameters are set.

        Raises ValueError if anything is missing or invalid.
        """
        problems = []

        p = self._parameters['period'].value
        pdot = self._parameters['p_dot'].value
        if np.isnan(p):
            problems.append(
                "period is not set. Use model.set_pulsar() or "
                "model.period.value = <Quantity>."
            )
        if np.isnan(pdot):
            problems.append(
                "p_dot is not set. Use model.set_pulsar() or "
                "model.p_dot.value = <float>."
            )

        if not problems:
            prm = self._parameters
            problems += check_physical_inputs(
                esn=prm['esn'].value, mej=prm['mej'].value,
                n_ism=prm['n_ism'].value,
                braking_index=prm['braking_index'].value,
                tau_sd=prm['tau_sd'].value, period=p, p_dot=pdot,
                eta_b=prm['eta_b'].value, eta_g=prm['eta_g'].value,
                e_min=prm['e_min'].value, e_max=prm['e_max'].value,
                e_break=prm['e_break'].value,
                p1=prm['p1'].value, p2=prm['p2'].value,
                f_max=prm['f_max'].value, vel_psr=prm['vel_psr'].value,
                kT_max=prm['kT_max'].value,
                distance=prm['distance'].value,
                n_ic=self.n_ic, ic_temps=self.ic_temps,
                ic_norms=self.ic_norms,
            )

        from .injection import INJECTION_MODELS
        if self.injection_model not in INJECTION_MODELS:
            problems.append(
                f"injection_model must be one of "
                f"{sorted(INJECTION_MODELS)} (or registered via "
                f"viper.injection.register_injection_model), got "
                f"{self.injection_model!r}"
            )

        if self.ssc_model not in ('off', 'uniform'):
            problems.append(
                f"ssc_model must be 'off' or 'uniform', got "
                f"{self.ssc_model!r}"
            )

        if problems:
            raise ValueError(
                "Model validation failed:\n  - " +
                "\n  - ".join(problems)
            )

    def apply_run_config(self, run_cfg: dict) -> None:
        """Apply model switches given as a dict (a source file's ``OPTIONS``).

        Consumes the evolution-level keys shared by every source:
        stepping options (``timestep_events``, ``snapshot_mode``,
        ``event_dt_floor_yr``, ``timestep_cooling``), escape keys
        (``escape_model``, ``esc_norm``, ``esc_index``,
        ``escape_fit``), ``injection_model``, and ``use_cmb_lookup``.
        Observables, priors, bounds, and params stay per-source.

        MUST be called BEFORE ``build_fit_spec``: the fit spec reads
        these from the model at build time.

        Absent keys keep the defaults (secant snapshot, no event
        stepper, compat cooling bound, escape off, lookup on).
        """
        from .escape import ESCAPE_MODELS

        # Stepping options
        self.timestep_events = bool(run_cfg.get('timestep_events', False))
        self.snapshot_mode = run_cfg.get('snapshot_mode', 'secant')
        self.event_dt_floor_yr = float(run_cfg.get('event_dt_floor_yr', 0.5))
        self.timestep_cooling = run_cfg.get('timestep_cooling', 'compat')
        if self.snapshot_mode not in ('secant', 'instantaneous'):
            raise ValueError(
                f"snapshot_mode must be 'secant' or "
                f"'instantaneous', got {self.snapshot_mode!r}")
        if self.timestep_cooling not in ('compat', 'total_loss10'):
            raise ValueError(
                f"timestep_cooling must be 'compat' or "
                f"'total_loss10', got {self.timestep_cooling!r}")

        # escape_fit bounds unfreeze esc_norm/esc_index.
        self.escape_model = run_cfg.get('escape_model', 'off')
        if self.escape_model not in ESCAPE_MODELS:
            raise ValueError(
                f"escape_model must be one of "
                f"{sorted(ESCAPE_MODELS)}, got {self.escape_model!r}")
        for name in ('esc_norm', 'esc_index'):
            if name in run_cfg:
                self._parameters[name].value = float(run_cfg[name])
        for name, bnds in run_cfg.get('escape_fit', {}).items():
            if name not in ('esc_norm', 'esc_index'):
                raise ValueError(
                    f"escape_fit key must be esc_norm or esc_index, "
                    f"got {name!r}")
            p_esc = self._parameters[name]
            p_esc.frozen = False
            p_esc.bounds = tuple(bnds)

        # Register custom shapes at import time.
        from .injection import INJECTION_MODELS
        self.injection_model = run_cfg.get(
            'injection_model', 'broken_powerlaw')
        if self.injection_model not in INJECTION_MODELS:
            raise ValueError(
                f"injection_model must be one of "
                f"{sorted(INJECTION_MODELS)} (or registered via "
                f"viper.injection.register_injection_model), got "
                f"{self.injection_model!r}")

        self.ssc_model = run_cfg.get('ssc_model', 'off')
        if self.ssc_model not in ('off', 'uniform'):
            raise ValueError(
                f"ssc_model must be 'off' or 'uniform', got "
                f"{self.ssc_model!r}")

        # Runner default True; constructor default False.
        self.use_cmb_lookup = bool(run_cfg.get('use_cmb_lookup', True))

    def apply_frozen_params(self, frozen_params: dict) -> None:
        """Freeze parameters at config-supplied values (INTERNAL units).

        Values are internal units (esn in 10^51 erg, distance in pc,
        energies in GeV), assigned via ``_set_internal``.

        Raises ValueError on an unknown parameter name.
        """
        for pname, pval in frozen_params.items():
            if pname not in self._parameters:
                raise ValueError(
                    f"frozen_params: unknown parameter '{pname}' "
                    f"(available: {self._parameters.names})")
            par = self._parameters[pname]
            par._set_internal(float(pval))
            par.frozen = True

    def run(self, dyn_step: int = 1, elec_step: int = -1,
            phot_step: int = -1, integration: str = 'pc_full',
            cpp_compat: bool = False,
            optimize_vtilda: bool = None,
            ic_early_break: bool = None,
            ic_j_break: bool = None,
            ic_skip_empty_bins: bool = True,
            escape_model: str = None,
            injection_model: str = None,
            ssc_model: str = None,
            snapshot_injection: str = 'included') -> PWNModelResult:
        """Run the full PWN evolution model.

        Parameters
        ----------
        dyn_step : int
            Cadence for dynamical snapshots (1=every step, -1=final only).
        elec_step : int
            Cadence for electron spectrum snapshots.
        phot_step : int
            Cadence for photon spectrum snapshots.
        integration : str
            PWN radius/velocity update: ``'euler'``, ``'pc_radius'``,
            or ``'pc_full'`` (predictor-corrector; default).
        cpp_compat : bool
            Force the original implementation's integration choices for
            bit-level reproduction.
        optimize_vtilda, ic_early_break, ic_j_break : bool, optional
            Performance/accuracy switches passed through to
            ``evolve_pwn``.
        ic_skip_empty_bins : bool
            Skip empty electron bins in the IC kernel (output unchanged).
        escape_model, injection_model, ssc_model : str, optional
            If given, assign the corresponding model attribute before
            running (persists on the model); ``ssc_model`` is ``'off'``
            or ``'uniform'``.
        snapshot_injection : {'included', 'excluded'}
            Snapshot content: with the step's own injection deposit
            (default) or the resident population only (display option;
            see ``evolve_pwn``).

        Returns
        -------
        PWNModelResult
            Dynamics, electron and photon spectra.
        """
        self.validate()
        p = self._parameters
        logger.info(
            "Running PWN model: age=%.1f yr, tau_sd=%.1f yr",
            self.true_age, p['tau_sd'].value,
        )
        if escape_model is not None:
            self.escape_model = escape_model
        if injection_model is not None:
            self.injection_model = injection_model
            self.validate()  # re-check name
        if ssc_model is not None:
            self.ssc_model = ssc_model
            self.validate()
        self._result = evolve_pwn(
            esn_51=p['esn'].value,
            mej=p['mej'].value,
            n_ism=p['n_ism'].value,
            braking_index=p['braking_index'].value,
            tau_sd=p['tau_sd'].value,
            age=self.true_age,
            e0_37=self.e0_37,
            vel_psr=p['vel_psr'].value,
            eta_g=p['eta_g'].value,
            eta_b=p['eta_b'].value,
            e_min=p['e_min'].value,
            e_max=p['e_max'].value,
            e_break=p['e_break'].value,
            p1=p['p1'].value,
            p2=p['p2'].value,
            f_max=p['f_max'].value,
            kT_max=p['kT_max'].value,
            n_ic=self.n_ic,
            ic_temps=self.ic_temps,
            ic_norms=self.ic_norms,
            n_elec_bins=self.n_elec_bins,
            n_phot_bins=self.n_phot_bins,
            n_ic_target=self.n_ic_target,
            phot_freq_min=self.phot_freq_min,
            phot_freq_max=self.phot_freq_max,
            target_freq_min=self.target_freq_min,
            target_freq_max=self.target_freq_max,
            dyn_step=dyn_step,
            elec_step=elec_step,
            phot_step=phot_step,
            integration=integration,
            cpp_compat=cpp_compat,
            optimize_vtilda=optimize_vtilda,
            ic_early_break=ic_early_break,
            ic_j_break=ic_j_break,
            ic_skip_empty_bins=ic_skip_empty_bins,
            use_cmb_lookup=self.use_cmb_lookup,
            escape_model=self.escape_model,
            esc_norm=p['esc_norm'].value,
            esc_index=p['esc_index'].value,
            injection_model=self.injection_model,
            ssc_model=self.ssc_model,
            snapshot_injection=snapshot_injection,
        )
        return self._result

    def run_for_fitting(self) -> PWNModelResult:
        """Run the model with settings optimized for MCMC (final step only).

        Returns PWNModelResult or raises on failure.
        """
        return self.run(dyn_step=-1)

    @property
    def result(self) -> Optional[PWNModelResult]:
        """Most recent model result, or None."""
        return self._result

    # Convenience: create from the legacy theta vector

    @classmethod
    def from_theta(
        cls,
        theta: Sequence[float],
        period: float,
        p_dot: float,
        distance: Optional[float] = None,
        ic_fields: Optional[Sequence[float]] = None,
    ) -> PWNModel:
        """Create from the 12-element theta vector (legacy interface).

        Parameters
        ----------
        theta : sequence of 12 floats
            ``[esn, mej, n_ism, distance, braking_index, tau_sd,
            eta_b, e_min, e_max, e_break, p1, p2]``, all in internal
            physical units (NOT log space).
        period : float
            Pulsar spin period [s].
        p_dot : float
            Period derivative [s/s].
        distance : float, optional
            Override distance from theta[3].
        ic_fields : sequence, optional
            [n_fields, T1, norm1, T2, norm2, ...].
        """
        if len(theta) < 12:
            raise ValueError(
                f"theta must have >= 12 elements, got {len(theta)}"
            )
        model = cls()
        # Internal units; bypass validation.
        model._parameters['esn']._set_internal(theta[0])
        model._parameters['mej']._set_internal(theta[1])
        model._parameters['n_ism']._set_internal(theta[2])
        model._parameters['distance']._set_internal(
            distance if distance is not None else theta[3]
        )
        model._parameters['braking_index']._set_internal(theta[4])
        model._parameters['tau_sd']._set_internal(theta[5])
        model._parameters['eta_b']._set_internal(theta[6])
        model._parameters['e_min']._set_internal(theta[7])
        model._parameters['e_max']._set_internal(theta[8])
        model._parameters['e_break']._set_internal(theta[9])
        model._parameters['p1']._set_internal(theta[10])
        model._parameters['p2']._set_internal(theta[11])
        model._parameters['period']._set_internal(period)
        model._parameters['p_dot']._set_internal(p_dot)

        # IC fields
        if ic_fields is not None and len(ic_fields) >= 3:
            n_ic = int(ic_fields[0])
            for i in range(n_ic):
                model.add_ic_field(
                    temp=ic_fields[1 + 2 * i],
                    norm=ic_fields[2 + 2 * i],
                )

        return model

    def copy(self) -> PWNModel:
        """Return a deep copy of this model."""
        return copy.deepcopy(self)

    # Display

    def __repr__(self) -> str:
        p = self._parameters['period'].value
        pdot = self._parameters['p_dot'].value
        p_str = f'{p:.6f} s' if not np.isnan(p) else 'not set'
        pdot_str = f'{pdot:.3e} s/s' if not np.isnan(pdot) else 'not set'
        nfree = self._parameters.ndim
        return (
            f"PWNModel(period={p_str}, p_dot={pdot_str}, "
            f"{nfree} free params, {self.n_ic} IC fields)"
        )
