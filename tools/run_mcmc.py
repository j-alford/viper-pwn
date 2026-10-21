#!/usr/bin/env python3
"""MCMC runner.

Usage:
    # New run (creates results/g54_run01, then _run02, ...):
    python3 tools/run_mcmc.py examples/g54.py

    # Add steps to an existing run:
    python3 tools/run_mcmc.py results/g54_run01 --extend 2000

The source file describes the fit (see examples/g54.py).  Each run
folder holds:

    source.py        copy of the source file as it was at launch
    chain.h5         emcee chain
    best_theta.npy   best point of the chain (fit space)
    mcmc.pid         process id while the run is active
"""
import os
# must precede viper import
os.environ.setdefault('NUMBA_NUM_THREADS', '1')

import argparse
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import emcee
from viper.fitting import (
    run_mcmc, log_probability, log_prior,
    _init_worker, _log_prob_worker,
)
from viper.source import (
    SourceError, build_from_source, load_run, load_source, new_run_dir,
    snapshot_path, source_settings, start_run,
)


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


def _chain_guard(h5file, outdir, overwrite):
    """Return True if a new run may write ``h5file``."""
    if not os.path.exists(h5file):
        return True
    try:
        steps_done = emcee.backends.HDFBackend(h5file, read_only=True).iteration
    except Exception as e:
        # unreadable chain must not bypass guard
        if not overwrite:
            print(f"Cannot read existing chain.h5 in {outdir} "
                  f"({type(e).__name__}: {e}).")
            print("Refusing to overwrite a chain that can't be "
                  "inspected. Pass --overwrite to discard it.")
            return False
        print(f"chain.h5 unreadable ({type(e).__name__}); "
              f"--overwrite given, discarding.")
        return True
    if steps_done >= 100 and not overwrite:
        print(f"chain.h5 already has {steps_done} steps in {outdir}.")
        print("Use --extend N to add more steps, or --overwrite to "
              "discard and restart.")
        return False
    if steps_done > 0:
        print(f"Overwriting chain with {steps_done} steps.")
    return True


def main():
    parser = argparse.ArgumentParser(description='PWN model MCMC runner')
    parser.add_argument('target',
                        help='Source file (new run) or run folder '
                             '(existing run)')
    parser.add_argument('--outdir', default=None,
                        help='Run folder for a new run '
                             '(default: results/<name>_runNN)')
    parser.add_argument('--extend', type=int, default=None,
                        help='Extend existing chain by N steps')
    parser.add_argument('--ncores', type=int, default=8)
    parser.add_argument('--overwrite', action='store_true',
                        help='Allow overwriting an existing chain (required if chain has >100 steps)')
    args = parser.parse_args()

    new_from_file = os.path.isfile(args.target)
    try:
        if new_from_file:
            if args.extend:
                print("--extend takes a run folder, not a source file.")
                return 2
            src = load_source(args.target)
            outdir = args.outdir or new_run_dir(args.target)
        else:
            if args.outdir:
                print("--outdir only applies when starting from a "
                      "source file.")
                return 2
            outdir = args.target
            src = load_run(outdir)
        model, obs, spec = build_from_source(src)
        fit = source_settings(src)['FIT']
    except SourceError as e:
        print(e)
        return 2

    h5file = os.path.join(outdir, 'chain.h5')
    # pid written after all guards
    pid_file = os.path.join(outdir, 'mcmc.pid')

    if not args.extend and not _chain_guard(h5file, outdir, args.overwrite):
        return 1

    print(f"{spec.ndim} free params: {spec.param_names}")
    print(f"{len(obs)} observables, integration={spec.integration}, "
          f"use_cmb_lookup={spec.use_cmb_lookup}")

    # --- Extend existing chain ---
    if args.extend:
        if not os.path.exists(h5file):
            print(f"No chain.h5 found in {outdir}")
            return 2
        os.environ['VIPER_SOURCE_FILE'] = os.path.abspath(snapshot_path(outdir))

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
        _save_best(sampler, outdir, spec)
        if os.path.exists(pid_file): os.remove(pid_file)
        return

    # --- New run ---
    nsteps = int(fit['nsteps'])
    nwalkers = int(fit['nwalkers'])
    if nwalkers < 2 * spec.ndim:
        print(f"FIT['nwalkers'] = {nwalkers} is too few for {spec.ndim} "
              f"free parameters; use at least {2 * spec.ndim}.")
        return 2

    # Build starting point
    default_p0 = np.array([model.parameters[fp.name].fit_value
                            for fp in spec.free_params])

    init_from = fit['init_from']
    if init_from:
        # never fall back to defaults
        if not os.path.exists(init_from):
            print(f"init_from file not found: {init_from}")
            return 2
        loaded = np.load(init_from)
        if len(loaded) != spec.ndim:
            print(f"init_from has {len(loaded)} params, "
                  f"the fit has {spec.ndim}. Fix FIT['init_from'] "
                  f"(or remove it).")
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

    scatter = fit['scatter']
    seed = int(fit['seed'])
    print(f"seed = {seed} (walker scatter + emcee proposals)")
    rng = np.random.default_rng(seed)
    scales = np.maximum(np.abs(p0_center) * scatter, 1e-3)
    p0 = p0_center + scales * rng.standard_normal((nwalkers, spec.ndim))
    for i, fp in enumerate(spec.free_params):
        lo, hi = fp.bounds
        p0[:, i] = np.clip(p0[:, i], lo + 1e-10, hi - 1e-10)
    # Walkers must start where the prior is finite (an AGE range can cut a
    # thin slice through the scatter ball); redraw the ones that do not.
    bad = np.array([not np.isfinite(log_prior(w, spec)) for w in p0])
    for _ in range(1000):
        if not bad.any():
            break
        p0[bad] = p0_center + scales * rng.standard_normal((bad.sum(), spec.ndim))
        for i, fp in enumerate(spec.free_params):
            lo, hi = fp.bounds
            p0[bad, i] = np.clip(p0[bad, i], lo + 1e-10, hi - 1e-10)
        bad = np.array([not np.isfinite(log_prior(w, spec)) for w in p0])
    if bad.any():
        print(f"{bad.sum()} of {nwalkers} walkers still start outside the prior "
              f"after 1000 redraws; reduce FIT['scatter'] or move the start.")
        return 2

    if new_from_file:
        start_run(args.target, outdir)
    print(f"Run folder: {outdir}")
    os.environ['VIPER_SOURCE_FILE'] = os.path.abspath(snapshot_path(outdir))

    _write_pid(pid_file)
    print(f"\nStarting MCMC: {nwalkers} walkers x {nsteps} steps, "
          f"{args.ncores} cores")
    t0 = time.time()
    sampler, spec = run_mcmc(
        model, obs, p0=p0, nwalkers=nwalkers, nsteps=nsteps,
        n_cores=args.ncores,
        moves=[(emcee.moves.StretchMove(a=1.3), 1.0)],
        integration=spec.integration,
        backend_file=h5file,
        seed=seed,
    )
    elapsed = time.time() - t0
    print(f"Done. {nsteps} steps in {elapsed/3600:.1f} hrs")
    _save_best(sampler, outdir, spec)
    if os.path.exists(pid_file): os.remove(pid_file)


if __name__ == '__main__':
    sys.exit(main() or 0)
