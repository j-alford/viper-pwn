#!/usr/bin/env python3
"""Unified MCMC runner for all sources.

Reads run_config.json from the output directory, selects the
correct source config based on the 'source' key (default: 'cta1').

Usage:
    # New run:
    python3 tools/run_mcmc.py results/sgr0501_run02

    # Resume/extend:
    python3 tools/run_mcmc.py results/sgr0501_run02 --extend 2000
"""
import os
# must precede viper import
os.environ.setdefault('NUMBA_NUM_THREADS', '1')

import argparse
import sys
import json
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import astropy.units as u
import emcee
from viper.fitting import (
    build_fit_spec, run_mcmc, log_probability, log_prior,
    _init_worker, _log_prob_worker,
)


# ----------------------------------------------------------------
# Source selection
# ----------------------------------------------------------------

def _source_module(source):
    """Import and return the source config module for ``source``."""
    if source == 'cta1':
        from examples import cta1_config as mod
    elif source == 'ctb87':
        from examples import ctb87_config as mod
    elif source == 'sgr0501':
        from examples import sgr0501_config as mod
    else:
        raise ValueError(
            f"Unknown source '{source}'. "
            f"Add a build_from_config() to examples/{source}_config.py "
            f"and register it in tools/run_mcmc.py")
    return mod


def _build_from_config(run_cfg):
    """Select the source config and return (model, obs, spec).

    Each source config module must provide:
        build_from_config(run_cfg) -> (model, obs, spec)
    """
    source = run_cfg.get('source', 'cta1')
    return _source_module(source).build_from_config(run_cfg)


# ----------------------------------------------------------------
# run_config validation (launch only; monitor stays lenient)
# ----------------------------------------------------------------

# Source-specific keys: EXTRA_RUN_CONFIG_KEYS in the source config module.
KNOWN_RUN_CONFIG_KEYS = frozenset({
    # runner (this file)
    'source', 'params', 'nsteps', 'nwalkers', 'scatter', 'seed',
    'init_from', 'use_cmb_lookup',
    # nested-sampling runner (same config format)
    'sampler', 'nlive', 'dlogz', 'slice_steps', 'resume',
    'n_cores', 'numba_threads',
    # PWNModel.apply_run_config
    'timestep_events', 'snapshot_mode', 'event_dt_floor_yr',
    'timestep_cooling', 'escape_model', 'esc_norm', 'esc_index',
    'escape_fit', 'injection_model', 'ssc_model',
    # shared per-source fit keys
    'frozen_params', 'mej_prior', 'distance_prior',
    'e_max_bounds', 'e_break_bounds', 'ic_fields', 'xray_band_limits',
    # tools/monitor.py display-path keys (historical runs carry them)
    'cxo_weight', 'cxo_scale', 'pwn_range', 'gamma_upper_limits',
})

# Annotation keys, ignored by code; '_'-prefixed keys likewise.
ANNOTATION_RUN_CONFIG_KEYS = frozenset({'description', 'notes', 'scenario'})


def validate_run_config(run_cfg):
    """Fail fast on unknown run_config keys."""
    mod = _source_module(run_cfg.get('source', 'cta1'))
    extra = frozenset(getattr(mod, 'EXTRA_RUN_CONFIG_KEYS', ()))
    allowed = KNOWN_RUN_CONFIG_KEYS | ANNOTATION_RUN_CONFIG_KEYS | extra
    unknown = sorted(k for k in run_cfg
                     if k not in allowed and not k.startswith('_'))
    if unknown:
        raise ValueError(
            f"unknown run_config key(s) {unknown}; typo'd keys would "
            f"silently fall back to defaults. Known keys: "
            f"{sorted(allowed)}. For human annotations use "
            f"description/notes/scenario or an '_'-prefixed key; for a "
            f"new source-specific key add it to EXTRA_RUN_CONFIG_KEYS "
            f"in the source config module.")


# ----------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------

def _write_pid(pid_file):
    with open(pid_file, 'w') as f:
        f.write(str(os.getpid()))
    print(f"PID {os.getpid()} written to {pid_file}")


def _save_best(sampler, outdir, spec):
    lps = sampler.get_log_prob()
    best_idx = np.unravel_index(np.argmax(lps), lps.shape)
    best_theta = sampler.get_chain()[best_idx]
    best_lp = lps[best_idx]
    np.save(os.path.join(outdir, 'best_theta.npy'), best_theta)
    # best_lp is posterior; chi2 excludes prior penalty
    lp_prior = log_prior(best_theta, spec) if np.isfinite(best_lp) else 0.0
    line = f"Best chi2 = {-2*(best_lp - lp_prior):.2f}"
    if lp_prior != 0.0:
        line += f" (+ prior dchi2 = {-2*lp_prior:.2f})"
    print(line)
    print(f"Saved best_theta.npy")


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='PWN model MCMC runner')
    parser.add_argument('outdir', help='Output directory (e.g. results/cta1_run15)')
    parser.add_argument('--extend', type=int, default=None,
                        help='Extend existing chain by N steps')
    parser.add_argument('--ncores', type=int, default=8)
    parser.add_argument('--overwrite', action='store_true',
                        help='Allow overwriting an existing chain (required if chain has >100 steps)')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    h5file = os.path.join(args.outdir, 'chain.h5')
    # pid written after all guards
    pid_file = os.path.join(args.outdir, 'mcmc.pid')
    config_file = os.path.join(args.outdir, 'run_config.json')

    # --- Overwrite guard ---
    if not args.extend and os.path.exists(h5file):
        try:
            existing = emcee.backends.HDFBackend(h5file, read_only=True)
            steps_done = existing.iteration
        except Exception as e:
            # unreadable chain must not bypass guard
            if not args.overwrite:
                print(f"Cannot read existing chain.h5 in {args.outdir} "
                      f"({type(e).__name__}: {e}).")
                print("Refusing to overwrite a chain that can't be "
                      "inspected. Pass --overwrite to discard it.")
                return 1
            print(f"chain.h5 unreadable ({type(e).__name__}); "
                  f"--overwrite given, discarding.")
            steps_done = None
        if steps_done is not None:
            if steps_done >= 100 and not args.overwrite:
                print(f"chain.h5 already has {steps_done} steps in {args.outdir}.")
                print("Use --extend N to add more steps, or --overwrite to discard and restart.")
                return 1
            elif steps_done > 0:
                print(f"Overwriting chain with {steps_done} steps.")

    # --- Read config ---
    if not os.path.exists(config_file):
        print(f"No run_config.json in {args.outdir}. Create one first.")
        return 2
    with open(config_file) as f:
        run_cfg = json.load(f)

    try:
        validate_run_config(run_cfg)
    except ValueError as e:
        print(f"run_config.json invalid: {e}")
        return 2

    model, obs, spec = _build_from_config(run_cfg)
    # defensive re-sync; --extend uses spec directly
    model.use_cmb_lookup = bool(run_cfg.get('use_cmb_lookup', True))
    spec.use_cmb_lookup = model.use_cmb_lookup
    print(f"use_cmb_lookup={model.use_cmb_lookup} "
          f"(default True for emcee; ~1.7x median speedup on CMB-only runs)")

    # --- Extend existing chain ---
    if args.extend:
        if not os.path.exists(h5file):
            print(f"No chain.h5 found in {args.outdir}")
            return 2

        backend = emcee.backends.HDFBackend(h5file)
        print(f"Extending {h5file}: {backend.iteration} steps done, "
              f"adding {args.extend}")

        from multiprocessing import Pool
        pool = Pool(args.ncores, initializer=_init_worker, initargs=(spec,))
        sampler = emcee.EnsembleSampler(
            backend.shape[0], spec.ndim,
            _log_prob_worker,
            backend=backend,
            pool=pool,
            moves=[(emcee.moves.StretchMove(a=1.3), 1.0)],
        )
        _write_pid(pid_file)
        t0 = time.time()
        sampler.run_mcmc(None, args.extend, progress=True)
        pool.close()
        pool.join()
        elapsed = time.time() - t0
        print(f"Done. {args.extend} steps in {elapsed/3600:.1f} hrs")
        _save_best(sampler, args.outdir, spec)
        if os.path.exists(pid_file): os.remove(pid_file)
        return

    # --- New run ---
    nsteps = run_cfg.get('nsteps', 1000)
    nwalkers = run_cfg.get('nwalkers', 96)

    # Build starting point
    default_p0 = np.array([model.parameters[fp.name].fit_value
                            for fp in spec.free_params])

    init_from = run_cfg.get('init_from', None)
    if init_from:
        # never fall back to defaults
        if not os.path.exists(init_from):
            print(f"init_from file not found: {init_from}")
            return 2
        loaded = np.load(init_from)
        if len(loaded) != spec.ndim:
            print(f"init_from has {len(loaded)} params, "
                  f"spec has {spec.ndim}. Fix init_from (or remove it) "
                  f"in run_config.json.")
            return 2
        p0_center = loaded
    else:
        p0_center = default_p0

    lp0 = log_probability(p0_center, spec)
    lpr0 = log_prior(p0_center, spec) if np.isfinite(lp0) else 0.0
    line = f"Starting point chi2 = {-2*(lp0 - lpr0):.2f}"
    if lpr0 != 0.0:
        line += f" (+ prior dchi2 = {-2*lpr0:.2f})"
    print(line)

    scatter = run_cfg.get('scatter', 5e-3)
    seed = int(run_cfg.get('seed', 42))
    print(f"seed = {seed} (walker scatter + emcee proposals)")
    rng = np.random.default_rng(seed)
    scales = np.maximum(np.abs(p0_center) * scatter, 1e-3)
    p0 = p0_center + scales * rng.standard_normal((nwalkers, spec.ndim))
    for i, fp in enumerate(spec.free_params):
        lo, hi = fp.bounds
        p0[:, i] = np.clip(p0[:, i], lo + 1e-10, hi - 1e-10)

    _write_pid(pid_file)
    print(f"\nStarting MCMC: {nwalkers} walkers x {nsteps} steps, "
          f"{args.ncores} cores")
    t0 = time.time()
    sampler, spec = run_mcmc(
        model, obs, p0=p0, nwalkers=nwalkers, nsteps=nsteps,
        n_cores=args.ncores,
        moves=[(emcee.moves.StretchMove(a=1.3), 1.0)],
        backend_file=h5file,
        seed=seed,
    )
    elapsed = time.time() - t0
    print(f"Done. {nsteps} steps in {elapsed/3600:.1f} hrs")
    _save_best(sampler, args.outdir, spec)
    if os.path.exists(pid_file): os.remove(pid_file)


if __name__ == '__main__':
    sys.exit(main() or 0)
