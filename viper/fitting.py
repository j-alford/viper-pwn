"""
MCMC fitting infrastructure for the PWN model.

The user-facing interface is ``run_mcmc(model, observables, ...)``.
The model's free (unfrozen) parameters define the MCMC dimensionality.
"""

from __future__ import annotations

import logging
import os
import time as _time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .evolution import evolve_pwn, PWNModelResult
from .observables import Observable, extract_all
from .pulsar import true_age_yr, tau_sd_from_age_yr, e0_37_from_timing
from . import constants as C

logger = logging.getLogger(__name__)


# Multiprocessing worker setup
_worker_spec = None

def _init_worker(spec):
    """Pool initializer: store spec once per worker process.

    Installs WARNING logging so eval-failure warnings reach the run log.
    """
    global _worker_spec
    _worker_spec = spec
    logging.basicConfig(level=logging.WARNING)
    # a source file may register custom injection shapes at import
    src = os.environ.get('VIPER_SOURCE_FILE')
    if src:
        from .source import load_source
        load_source(src)

def _log_prob_worker(theta):
    """Thin wrapper that reads spec from the worker global."""
    return log_probability(theta, _worker_spec)

def _log_like_worker(theta):
    """Worker for nested sampling: bare log_likelihood (NS handles the prior
    via prior_transform, so there's no log_prior contribution here)."""
    return log_likelihood(theta, _worker_spec)



# Internal fit specification

@dataclass
class _ParamSpec:
    """Specification of a single free parameter for fitting."""
    name: str
    log: bool
    bounds: tuple[float, float]
    prior: Optional[tuple[float, float]]  # (mu, sigma); physical space unless prior_log
    prior_log: bool = False  # Gaussian applied to theta directly


@dataclass
class _FitSpec:
    """Complete pickle-safe snapshot of a fitting problem.

    Created once by ``run_mcmc`` from the model + observables.
    Passed as the extra argument to ``log_probability``.
    """
    # Free parameter definitions
    free_params: list[_ParamSpec]
    # All parameter values (frozen + free defaults)
    all_values: dict[str, float]
    # Derived quantities that depend on parameters
    period: float
    p_dot: float
    # IC fields
    n_ic: int
    ic_temps: list[float]
    ic_norms: list[float]
    ic_temp_names: list[str]
    ic_norm_names: list[str]
    # Grid settings
    n_elec_bins: int = C.N_ELEC_BINS
    n_phot_bins: int = C.N_PHOT_FREQ
    n_ic_target: int = C.N_IC_TARGET
    phot_freq_min: float = C.PHOT_FREQ_MIN
    phot_freq_max: float = C.PHOT_FREQ_MAX
    target_freq_min: float = C.TARGET_FREQ_MIN
    target_freq_max: float = C.TARGET_FREQ_MAX
    # Observables
    observables: list[Observable] = field(default_factory=list)
    # Evolution settings
    integration: str = 'pc_full'
    use_cmb_lookup: bool = False
    # Stepping options
    timestep_events: bool = False
    snapshot_mode: str = 'secant'
    event_dt_floor_yr: float = 0.5
    timestep_cooling: str = 'compat'
    resolve_final_cooling: bool = False
    # Escape toggle; esc_norm/esc_index in all_values
    escape_model: str = 'off'
    # Custom shapes: register at module level; spawned workers re-import.
    injection_model: str = 'broken_powerlaw'
    # SSC toggle; no new params
    ssc_model: str = 'off'
    known_age: float | None = None

    @property
    def ndim(self) -> int:
        return len(self.free_params)

    @property
    def param_names(self) -> list[str]:
        return [p.name for p in self.free_params]

    @property
    def param_labels(self) -> list[str]:
        return [
            f'log({p.name})' if p.log else p.name
            for p in self.free_params
        ]


def _build_fit_spec(model, observables: list[Observable],
                    integration: str = 'pc_full') -> _FitSpec:
    """Extract a pickle-safe _FitSpec from a PWNModel."""
    from .model import PWNModel

    free_params = []
    for p in model.parameters.free:
        fb = p.fit_bounds
        if fb[0] == -np.inf or fb[1] == np.inf:
            raise ValueError(
                f"Free parameter '{p.name}' must have finite bounds. "
                f"Set model.{p.name}.bounds = (...) before fitting."
            )
        free_params.append(_ParamSpec(
            name=p.name,
            log=p.log,
            bounds=fb,
            prior=p.prior,
            prior_log=getattr(p, 'prior_log', False),
        ))

    if not free_params:
        raise ValueError("No free parameters. Unfreeze at least one parameter.")

    known_age = getattr(model, 'known_age', None)
    if known_age is not None and not model.parameters['tau_sd'].frozen:
        raise ValueError(
            "tau_sd follows from the braking index when the age is known; "
            "it cannot be a free parameter.")
    all_values = {p.name: p.value for p in model.parameters}

    ic_temp_names = [f'ic_temp_{i+1}' for i in range(model.n_ic)]
    ic_norm_names = [f'ic_norm_{i+1}' for i in range(model.n_ic)]

    return _FitSpec(
        free_params=free_params,
        all_values=all_values,
        period=model.parameters['period'].value,
        p_dot=model.parameters['p_dot'].value,
        n_ic=model.n_ic,
        ic_temps=list(model.ic_temps),
        ic_norms=list(model.ic_norms),
        ic_temp_names=ic_temp_names,
        ic_norm_names=ic_norm_names,
        n_elec_bins=model.n_elec_bins,
        n_phot_bins=model.n_phot_bins,
        n_ic_target=model.n_ic_target,
        phot_freq_min=model.phot_freq_min,
        phot_freq_max=model.phot_freq_max,
        target_freq_min=model.target_freq_min,
        target_freq_max=model.target_freq_max,
        observables=list(observables),
        integration=integration,
        use_cmb_lookup=getattr(model, 'use_cmb_lookup', False),
        timestep_events=bool(getattr(model, 'timestep_events', False)),
        snapshot_mode=getattr(model, 'snapshot_mode', 'secant'),
        event_dt_floor_yr=float(getattr(model, 'event_dt_floor_yr', 0.5)),
        timestep_cooling=getattr(model, 'timestep_cooling', 'compat'),
        resolve_final_cooling=bool(getattr(model, 'resolve_final_cooling', False)),
        escape_model=getattr(model, 'escape_model', 'off'),
        injection_model=getattr(model, 'injection_model',
                                'broken_powerlaw'),
        ssc_model=getattr(model, 'ssc_model', 'off'),
        known_age=known_age,
    )


def _theta_to_evolve_args(theta: np.ndarray, spec: _FitSpec) -> dict:
    """Convert MCMC theta + frozen values into evolve_pwn keyword args.

    This is the hot path: called once per likelihood evaluation.
    """
    vals = dict(spec.all_values)

    for i, fp in enumerate(spec.free_params):
        val = theta[i]
        if fp.log:
            val = 10.0 ** val
        vals[fp.name] = val

    ic_temps = [vals.get(n, 0.0) for n in spec.ic_temp_names]
    ic_norms = [vals.get(n, 0.0) for n in spec.ic_norm_names]

    # Derived quantities single-sourced in pulsar.py
    period = vals['period']
    p_dot = vals['p_dot']
    braking_index = vals['braking_index']
    if spec.known_age is not None:
        tau_sd = tau_sd_from_age_yr(period, p_dot, braking_index,
                                    spec.known_age)
        true_age = spec.known_age
    else:
        tau_sd = vals['tau_sd']
        true_age = true_age_yr(period, p_dot, braking_index, tau_sd)
    e0_37 = e0_37_from_timing(period, p_dot, braking_index, tau_sd)

    return dict(
        esn_51=vals['esn'],
        mej=vals['mej'],
        n_ism=vals['n_ism'],
        braking_index=braking_index,
        tau_sd=tau_sd,
        age=true_age,
        e0_37=e0_37,
        vel_psr=vals['vel_psr'],
        eta_g=vals['eta_g'],
        eta_b=vals['eta_b'],
        e_min=vals['e_min'],
        e_max=vals['e_max'],
        e_break=vals['e_break'],
        p1=vals['p1'],
        p2=vals['p2'],
        f_max=vals['f_max'],
        kT_max=vals['kT_max'],
        n_ic=spec.n_ic,
        ic_temps=ic_temps,
        ic_norms=ic_norms,
        n_elec_bins=spec.n_elec_bins,
        n_phot_bins=spec.n_phot_bins,
        n_ic_target=spec.n_ic_target,
        phot_freq_min=spec.phot_freq_min,
        phot_freq_max=spec.phot_freq_max,
        target_freq_min=spec.target_freq_min,
        target_freq_max=spec.target_freq_max,
        dyn_step=-1,
        integration=spec.integration,
        use_cmb_lookup=spec.use_cmb_lookup,
        timestep_events=spec.timestep_events,
        snapshot_mode=spec.snapshot_mode,
        event_dt_floor_yr=spec.event_dt_floor_yr,
        timestep_cooling=spec.timestep_cooling,
        resolve_final_cooling=spec.resolve_final_cooling,
        escape_model=spec.escape_model,
        esc_norm=vals.get('esc_norm', 1.0),
        esc_index=vals.get('esc_index', 0.0),
        injection_model=spec.injection_model,
        ssc_model=spec.ssc_model,
    )


# Log-probability components

# Bounded once-per-process warning dedup
_FAILURE_WARNED: set = set()
_FAILURE_WARNED_MAX = 100


def _note_eval_failure(stage: str, exc: Exception) -> None:
    """Log a likelihood-evaluation failure that is converted to -inf.

    First occurrence per (stage, type, message) logs at WARNING;
    repeats at DEBUG.
    """
    key = (stage, type(exc).__name__, str(exc)[:200])
    if key not in _FAILURE_WARNED and len(_FAILURE_WARNED) < _FAILURE_WARNED_MAX:
        _FAILURE_WARNED.add(key)
        logger.warning(
            "log_likelihood: %s raised %s: %s -> -inf (repeat "
            "occurrences logged at DEBUG)",
            stage, type(exc).__name__, exc)
    else:
        logger.debug("log_likelihood: %s raised %r -> -inf", stage, exc)


def log_prior(theta: np.ndarray, spec: _FitSpec) -> float:
    """Compute log-prior for a parameter vector.

    Flat (uniform) within bounds, with optional Gaussian terms.
    Returns -inf if any parameter is outside its bounds.
    ``prior_log=True`` applies the Gaussian to theta directly; otherwise
    to the PHYSICAL value (``10**theta`` for log-sampled parameters).
    Measure difference: this acts on emcee's flat-in-theta measure;
    ``nested.prior_transform`` draws the same prior in physical space.
    """
    lp = 0.0
    for i, fp in enumerate(spec.free_params):
        lo, hi = fp.bounds
        if theta[i] < lo or theta[i] > hi:
            return -np.inf
        if fp.prior is not None:
            mu, sigma = fp.prior
            if fp.prior_log:
                val = theta[i]
            else:
                val = 10**theta[i] if fp.log else theta[i]
            lp += -0.5 * ((val - mu) / sigma) ** 2
    return lp


def log_likelihood(theta: np.ndarray, spec: _FitSpec) -> float:
    """Compute log-likelihood: -1/2 chi2 comparing model to data.

    Returns -inf if the model crashes (unphysical parameter combo) or
    predicts NaN for a flux observable.
    """
    try:
        kw = _theta_to_evolve_args(theta, spec)
    except (ValueError, ZeroDivisionError):
        return -np.inf

    if kw['age'] <= 0:
        return -np.inf

    try:
        result = evolve_pwn(**kw)
    except Exception as exc:
        _note_eval_failure('evolve_pwn', exc)
        return -np.inf

    dyn = result.dyninfo[-1]
    vals = dict(spec.all_values)
    for i, fp in enumerate(spec.free_params):
        val = theta[i]
        if fp.log:
            val = 10.0 ** val
        vals[fp.name] = val
    distance_pc = vals['distance']

    try:
        predicted = extract_all(
            spec.observables,
            result.photon_freqs, result.photon_spectrum,
            dyn.rad_pwn, dyn.rad_snr, distance_pc,
        )
    except Exception as exc:
        _note_eval_failure('extract_all', exc)
        return -np.inf

    if np.any(np.isnan(predicted)):
        for k, obs in enumerate(spec.observables):
            if obs.obs_type.value == 'spectral_index':
                continue
            if np.isnan(predicted[k]):
                return -np.inf

    chi2 = 0.0
    for k, obs in enumerate(spec.observables):
        w = obs.weight
        if obs.range_lo is not None and obs.range_hi is not None:
            # Flat-top: zero penalty inside range
            if predicted[k] < obs.range_lo:
                chi2 += w * ((predicted[k] - obs.range_lo) / obs.error) ** 2
            elif predicted[k] > obs.range_hi:
                chi2 += w * ((predicted[k] - obs.range_hi) / obs.error) ** 2
            # else: inside range, no contribution
        elif obs.upper_limit:
            if predicted[k] > obs.value:
                chi2 += w * ((predicted[k] - obs.value) / obs.limit_sigma) ** 2
        elif obs.lower_limit:
            if predicted[k] < obs.value:
                chi2 += w * ((predicted[k] - obs.value) / obs.limit_sigma) ** 2
        else:
            if (obs.obs_type.value == 'spectral_index'
                    and np.isnan(predicted[k])):
                # NaN index: unconstrained, skip term (not excluded)
                continue
            chi2 += w * ((predicted[k] - obs.value) / obs.error) ** 2
    return -0.5 * chi2


def log_probability(theta: np.ndarray, spec: _FitSpec) -> float:
    """Log-posterior = log-prior + log-likelihood.

    This is the function passed to emcee as the target.
    """
    lp = log_prior(theta, spec)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, spec)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


# MCMC runner

def run_mcmc(
    model,
    observables: list[Observable],
    p0: Optional[np.ndarray] = None,
    nwalkers: int = 32,
    nsteps: int = 5000,
    n_cores: int = 1,
    progress: bool = True,
    moves: Optional[object] = None,
    integration: str = 'pc_full',
    backend_file: Optional[str] = None,
    seed: Optional[int] = None,
) -> tuple:
    """Run emcee MCMC on a PWNModel.

    Parameters
    ----------
    model : PWNModel
        Model with free parameters (unfrozen, with bounds set).
    observables : list of Observable
        Measured quantities to fit against.
    p0 : ndarray of shape (nwalkers, ndim), optional
        Initial walker positions in fit space; default Gaussian ball
        around current fit values.
    nwalkers : int
        Number of walkers (must be >= 2 * ndim).
    nsteps : int
        Total number of MCMC steps.
    n_cores : int
        CPU cores for parallel evaluation (1 = serial).
    progress : bool
        Show progress bar.
    moves : optional
        Custom emcee moves.
    backend_file : str, optional
        Path to HDF5 file for incremental chain storage.
    seed : int, optional
        Seed for walker initialization (ignored when ``p0`` is given)
        and emcee's proposal RNG; None = nondeterministic.

    Returns
    -------
    sampler : emcee.EnsembleSampler
        The sampler with the full chain.
    spec : _FitSpec
        The fit specification (needed by post-processing helpers).
    """
    try:
        import emcee
    except ImportError:
        raise ImportError(
            "emcee is required for MCMC fitting. Install with:\n"
            "  pip install emcee"
        )

    model.validate()

    spec = _build_fit_spec(model, observables, integration=integration)
    ndim = spec.ndim

    if nwalkers < 2 * ndim:
        nwalkers = 2 * ndim
        logger.warning("nwalkers increased to %d (2 x ndim)", nwalkers)

    if p0 is None:
        p0 = _init_walkers(spec, model, nwalkers,
                           rng=np.random.default_rng(seed))

    logger.info(
        "Starting MCMC: %d walkers x %d steps, %d params, %d observables",
        nwalkers, nsteps, ndim, len(observables),
    )
    if seed is not None:
        logger.info("Seed: %d (walker init + emcee proposals)", seed)
    logger.info("Parameters: %s", spec.param_labels)
    if n_cores > 1:
        logger.info("Using %d cores for parallel evaluation", n_cores)

    backend = None
    if backend_file is not None:
        import os
        backend = emcee.backends.HDFBackend(backend_file)
        backend.reset(nwalkers, ndim)
        logger.info('HDF5 backend: %s', os.path.abspath(backend_file))

    t0 = _time.perf_counter()

    sampler_kwargs = dict(
        nwalkers=nwalkers, ndim=ndim,
        log_prob_fn=log_probability, args=(spec,),
    )
    if backend is not None:
        sampler_kwargs['backend'] = backend
    if moves is not None:
        sampler_kwargs['moves'] = moves

    if n_cores > 1:
        from multiprocessing import Pool
        with Pool(n_cores, initializer=_init_worker, initargs=(spec,)) as pool:
            sampler_kwargs_par = dict(
                nwalkers=nwalkers, ndim=ndim,
                log_prob_fn=_log_prob_worker,
                pool=pool,
            )
            if backend is not None:
                sampler_kwargs_par['backend'] = backend
            if moves is not None:
                sampler_kwargs_par['moves'] = moves
            sampler = emcee.EnsembleSampler(**sampler_kwargs_par)
            if seed is not None:
                sampler.random_state = \
                    np.random.mtrand.RandomState(seed).get_state()
            sampler.run_mcmc(p0, nsteps, progress=progress)
    else:
        sampler = emcee.EnsembleSampler(**sampler_kwargs)
        if seed is not None:
            sampler.random_state = \
                np.random.mtrand.RandomState(seed).get_state()
        sampler.run_mcmc(p0, nsteps, progress=progress)

    t1 = _time.perf_counter()
    elapsed = t1 - t0
    logger.info(
        "MCMC complete in %.1f s (%.3f s/eval)",
        elapsed, elapsed / (nwalkers * nsteps),
    )
    af = sampler.acceptance_fraction
    logger.info("Mean acceptance fraction: %.3f (target: 0.2-0.5)", af.mean())

    return sampler, spec


def _init_walkers(
    spec: _FitSpec, model, nwalkers: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Initialize walkers in a small ball around current parameter values.

    ``rng`` makes the initialization reproducible (threaded from
    ``run_mcmc(seed=...)``); None draws a fresh Generator.
    """
    if rng is None:
        rng = np.random.default_rng()
    ndim = spec.ndim
    p0 = np.empty((nwalkers, ndim))

    for i, fp in enumerate(spec.free_params):
        param = model.parameters[fp.name]
        center = param.fit_value

        lo, hi = fp.bounds
        if center < lo or center > hi or not np.isfinite(center):
            center = 0.5 * (lo + hi)

        spread = 0.01 * (hi - lo)
        p0[:, i] = center + spread * rng.standard_normal(nwalkers)
        p0[:, i] = np.clip(p0[:, i], lo + 1e-10, hi - 1e-10)

    return p0


# Post-processing helpers

def get_best_fit(
    sampler, spec: _FitSpec, burn_in: int = 500
) -> dict[str, float]:
    """Extract the maximum-posterior (MAP) parameter set from the chain.

    With Gaussian priors active this is the MAP point, not the
    maximum-likelihood one; ``chi2`` = -2*log_likelihood has the
    prior penalty removed.
    Returns a dict mapping parameter names to physical values.
    """
    flat_chain = sampler.get_chain(discard=burn_in, flat=True)
    flat_log_prob = sampler.get_log_prob(discard=burn_in, flat=True)
    best_idx = np.argmax(flat_log_prob)
    best_theta = flat_chain[best_idx]

    result = {}
    for i, fp in enumerate(spec.free_params):
        val = best_theta[i]
        if fp.log:
            result[fp.name] = 10.0 ** val
            result[f'log_{fp.name}'] = val
        else:
            result[fp.name] = val
    lp = flat_log_prob[best_idx]
    lp_prior = log_prior(best_theta, spec)
    result['log_probability'] = lp
    result['log_prior'] = lp_prior
    result['log_likelihood'] = lp - lp_prior
    result['chi2'] = -2.0 * (lp - lp_prior)
    return result


def predict_observables(
    theta: np.ndarray, spec: _FitSpec
) -> tuple[np.ndarray, PWNModelResult]:
    """Run the model for a given theta and return predicted observables.

    Returns
    -------
    predicted : ndarray
        Predicted values for each observable.
    result : PWNModelResult
        Full model output.
    """
    kw = _theta_to_evolve_args(theta, spec)
    result = evolve_pwn(**kw)
    dyn = result.dyninfo[-1]

    vals = dict(spec.all_values)
    for i, fp in enumerate(spec.free_params):
        val = theta[i]
        if fp.log:
            val = 10.0 ** val
        vals[fp.name] = val

    predicted = extract_all(
        spec.observables,
        result.photon_freqs, result.photon_spectrum,
        dyn.rad_pwn, dyn.rad_snr, vals['distance'],
    )
    return predicted, result


def print_residuals(theta: np.ndarray, spec: _FitSpec) -> None:
    """Run model at theta and print a residual table."""
    import time as _t
    t0 = _t.time()
    pred, result = predict_observables(theta, spec)
    dt = _t.time() - t0
    dyn = result.dyninfo[-1]

    lp = log_probability(theta, spec)
    lp_prior = log_prior(theta, spec) if np.isfinite(lp) else 0.0
    chi2 = -2 * (lp - lp_prior) if np.isfinite(lp) else np.inf

    print("=" * 70)
    print("Model evaluation")
    print("=" * 70)
    line = f"\nlog_probability = {lp:.2f}  (chi2 = {chi2:.2f}"
    if lp_prior != 0.0:
        line += f", prior dchi2 = {-2 * lp_prior:.2f}"
    print(line + ")")
    print(f"Model eval time: {dt:.2f} s")
    print(f"\nDynamics: R_pwn = {dyn.rad_pwn:.2f} pc, "
          f"R_snr = {dyn.rad_snr:.2f} pc, B = {dyn.bpwn:.3f} uG")
    any_weights = any(o.weight != 1.0 for o in spec.observables)
    any_priors = any(fp.prior is not None for fp in spec.free_params)
    if any_weights:
        print(f"\n{'Observable':<35} {'Predicted':>12} {'Data':>12} "
              f"{'sigma':>12} {'d/sigma':>8} {'w':>4} {'w*chi2':>8}")
        print("-" * 98)
    else:
        print(f"\n{'Observable':<35} {'Predicted':>12} {'Data':>12} "
              f"{'sigma':>12} {'d/sigma':>8}")
        print("-" * 78)
    for k, obs in enumerate(spec.observables):
        if obs.range_lo is not None and obs.range_hi is not None:
            if pred[k] < obs.range_lo:
                dsig = (obs.range_lo - pred[k]) / obs.error
                wchi2 = obs.weight * dsig**2
                tail = f" {obs.weight:>4.1f} {wchi2:>8.2f}" if any_weights else ""
                print(f"{obs.label:<35} {pred[k]:>12.4g} "
                      f"[{obs.range_lo:.1f},{obs.range_hi:.1f}] "
                      f"{obs.error:>7.1f} {dsig:>+8.2f}" + tail)
            elif pred[k] > obs.range_hi:
                dsig = (obs.range_hi - pred[k]) / obs.error
                wchi2 = obs.weight * dsig**2
                tail = f" {obs.weight:>4.1f} {wchi2:>8.2f}" if any_weights else ""
                print(f"{obs.label:<35} {pred[k]:>12.4g} "
                      f"[{obs.range_lo:.1f},{obs.range_hi:.1f}] "
                      f"{obs.error:>7.1f} {dsig:>+8.2f}" + tail)
            else:
                tail = f" {obs.weight:>4.1f} {'0.00':>8}" if any_weights else ""
                print(f"{obs.label:<35} {pred[k]:>12.4g} "
                      f"[{obs.range_lo:.1f},{obs.range_hi:.1f}] "
                      f"{'---':>7} {'OK':>8}" + tail)
        elif obs.upper_limit:
            status = "OK" if pred[k] <= obs.value else "EXCEEDS"
            tail = "" if not any_weights else f" {'':>4} {'':>8}"
            print(f"{obs.label:<35} {pred[k]:>12.4g} "
                  f"{obs.value:>12.4g} {'---':>12} {status:>8}" + tail)
        elif obs.lower_limit:
            status = "OK" if pred[k] >= obs.value else "BELOW"
            tail = "" if not any_weights else f" {'':>4} {'':>8}"
            print(f"{obs.label:<35} {pred[k]:>12.4g} "
                  f">{obs.value:>11.4g} {'---':>12} {status:>8}" + tail)
        else:
            dsig = (obs.value - pred[k]) / obs.error
            wchi2 = obs.weight * dsig**2
            tail = f" {obs.weight:>4.1f} {wchi2:>8.2f}" if any_weights else ""
            print(f"{obs.label:<35} {pred[k]:>12.4g} "
                  f"{obs.value:>12.4g} {obs.error:>12.4g} {dsig:>+8.2f}" + tail)

    if any_weights or any_priors:
        print()
        obs_chi2_w = 0.0
        obs_chi2_uw = 0.0
        for k, obs in enumerate(spec.observables):
            if obs.upper_limit:
                continue
            if obs.lower_limit:
                if pred[k] >= obs.value:
                    continue
                dsig = (pred[k] - obs.value) / obs.error
                obs_chi2_w += obs.weight * dsig**2
                obs_chi2_uw += dsig**2
                continue
            if obs.range_lo is not None and obs.range_hi is not None:
                if obs.range_lo <= pred[k] <= obs.range_hi:
                    continue
                dsig = ((pred[k] - obs.range_lo) / obs.error if pred[k] < obs.range_lo
                        else (pred[k] - obs.range_hi) / obs.error)
            else:
                dsig = (pred[k] - obs.value) / obs.error
            obs_chi2_w += obs.weight * dsig**2
            obs_chi2_uw += dsig**2
        prior_chi2 = 0.0
        for fp in spec.free_params:
            if fp.prior is not None:
                center, sigma = fp.prior
                idx = spec.param_names.index(fp.name)
                val = 10**theta[idx] if fp.log else theta[idx]
                term = ((val - center) / sigma)**2
                prior_chi2 += term
                print(f"  Prior: {fp.name:<15s} val={val:.2f} "
                      f"prior=({center},{sigma})  chi2={term:.2f}")
        ndof = len(spec.observables) - len(spec.free_params)
        red_chi2 = obs_chi2_uw / ndof if ndof > 0 else 0.0
        print(f"  Obs chi2 = {obs_chi2_uw:.2f}, chi2/dof = {obs_chi2_uw:.2f}/{ndof} = {red_chi2:.2f}")
        if any_weights:
            print(f"  Weighted obs chi2 = {obs_chi2_w:.2f}, "
                  f"Prior chi2 = {prior_chi2:.2f}, "
                  f"Total (-2lnL) = {obs_chi2_w + prior_chi2:.2f}")


def decode_theta(theta: np.ndarray, spec: _FitSpec,
                 print_table: bool = True) -> dict:
    """Decode an MCMC theta vector to physical parameter values.

    Parameters
    ----------
    theta : array-like
        MCMC parameter vector.
    spec : _FitSpec
        Fit specification.
    print_table : bool
        If True, also print a formatted table to stdout.

    Returns
    -------
    dict
        Mapping of parameter name to physical value.
    """
    out = {}
    rows = []
    for i, fp in enumerate(spec.free_params):
        mcmc_val = theta[i]
        phys_val = 10.0 ** mcmc_val if fp.log else mcmc_val
        space = 'log' if fp.log else 'lin'
        rows.append((fp.name, mcmc_val, phys_val, space))
        out[fp.name] = phys_val
    if print_table:
        print(f"{'Parameter':<20s} {'MCMC value':>12s} {'Physical':>14s} {'Space':>6s}")
        print("-" * 56)
        for name, mcmc, phys, space in rows:
            print(f"{name:<20s} {mcmc:>12.4f} {phys:>14.4g} {space:>6s}")
    return out


# Convenience: build spec from model

def build_fit_spec(model, observables: list[Observable],
                   integration: str = 'pc_full') -> _FitSpec:
    """Build a fit specification from a model and observables.

    This is useful for calling log_probability, predict_observables,
    etc. outside of run_mcmc.

    Returns
    -------
    spec : _FitSpec
        Pickle-safe fit specification.
    """
    return _build_fit_spec(model, observables, integration=integration)
