#!/usr/bin/env python3
"""Nested-sampling runner (UltraNest).

Usage
-----
    # New run (creates results/g54_run01, then _run02, ...):
    python3 tools/run_nested.py examples/g54.py --ncores 4

    # Re-run the run folder to resume an interrupted run:
    python3 tools/run_nested.py results/g54_run01 --ncores 4

The source file describes the fit (see examples/g54.py).  Its optional
NESTED block sets:

    'nlive'        : int, default 800 (UltraNest min_num_live_points).
    'dlogz'        : float, default 0.5 (convergence target).
    'slice_steps'  : int or None, default None -> max(2*ndim, 8).
    'resume'       : 'resume' (default) or 'overwrite'.

Output layout (results/<run>/)
------------------------------
    source.py           copy of the source file as it was at launch
    ultranest/          UltraNest's native log_dir (resume support)
    posterior.npy       equal-weighted samples (fit space)
    chain.h5            emcee.HDFBackend-readable for monitor.py
    evidence.json       summary (logz, logzerr, ncall, etc.)
    best_theta.npy      max-logL fit-space point
    ns.pid              process id (parallel to mcmc.pid)
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..'))   # repo root: viper


def _early_set_numba_threads():
    """Set NUMBA_NUM_THREADS from the command line BEFORE viper imports."""
    early = argparse.ArgumentParser(add_help=False)
    early.add_argument('--ncores', type=int, default=1)
    early.add_argument('--numba-threads', type=int, default=None)
    known, _ = early.parse_known_args()
    n_threads = known.numba_threads
    if n_threads is None:
        n_threads = max(1, (os.cpu_count() or 16) // max(known.ncores, 1))
    os.environ.setdefault('NUMBA_NUM_THREADS', str(int(n_threads)))


_early_set_numba_threads()

from viper.nested import run_nested, save_nested_results  # noqa: E402
from viper.source import (  # noqa: E402
    SourceError, build_from_source, load_run, load_source, new_run_dir,
    snapshot_path, source_settings, start_run,
)


def _check_overwrite(outdir, overwrite):
    """Refuse to start if evidence.json exists and overwrite not set."""
    ev = os.path.join(outdir, 'evidence.json')
    if os.path.exists(ev) and not overwrite:
        print(f"evidence.json already present in {outdir}.")
        print("Pass --overwrite to discard and restart. An interrupted "
              "run (no evidence.json yet) resumes by default.")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description='PWN model nested-sampling runner')
    parser.add_argument('target',
                        help='Source file (new run) or run folder '
                             '(existing run)')
    parser.add_argument('--outdir', default=None,
                        help='Run folder for a new run '
                             '(default: results/<name>_runNN)')
    parser.add_argument('--ncores', type=int, default=1,
                        help='Number of worker processes for vectorized '
                             'likelihood (Pool size).  1 = serial.')
    parser.add_argument('--numba-threads', type=int, default=None,
                        help='Threads per process '
                             '(default: cpu_count // ncores).')
    parser.add_argument('--overwrite', action='store_true',
                        help='Allow overwriting an existing evidence.json.')
    args = parser.parse_args()

    new_from_file = os.path.isfile(args.target)
    try:
        if new_from_file:
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
        nested = source_settings(src)['NESTED']
    except SourceError as e:
        print(e)
        return 2

    if not _check_overwrite(outdir, args.overwrite):
        return 1

    nlive = int(nested['nlive'])
    dlogz = float(nested['dlogz'])
    slice_steps = nested['slice_steps']
    if slice_steps is not None:
        slice_steps = int(slice_steps)
    resume = nested['resume']
    n_cores = int(args.ncores)

    if new_from_file:
        start_run(args.target, outdir)
    print(f"Run folder: {outdir}")
    os.environ['VIPER_SOURCE_FILE'] = os.path.abspath(snapshot_path(outdir))

    pid_file = os.path.join(outdir, 'ns.pid')
    with open(pid_file, 'w') as f:
        f.write(str(os.getpid()))
    print(f"PID {os.getpid()} written to {pid_file}")

    log_dir = os.path.join(outdir, 'ultranest')

    print(f"{spec.ndim} free params: {spec.param_names}")
    print(f"{len(obs)} observables, integration={spec.integration}, "
          f"use_cmb_lookup={spec.use_cmb_lookup}")

    print(f"Starting NS: nlive={nlive}, dlogz={dlogz}, "
          f"slice_steps={slice_steps}, n_cores={n_cores}, resume={resume}, "
          f"log_dir={log_dir}")
    t0 = time.time()
    result, spec = run_nested(
        model, obs,
        sampler='ultranest',
        integration=spec.integration,
        nlive=nlive,
        dlogz=dlogz,
        slice_steps=slice_steps,
        n_cores=n_cores,
        log_dir=log_dir,
        resume=resume,
        show_status=True,
    )
    elapsed = time.time() - t0
    print(f"Done. {result.get('ncall', 0)} likelihood calls in "
          f"{elapsed/3600:.2f} hrs.  logz = {result.get('logz', float('nan')):.3f} "
          f"+- {result.get('logzerr', float('nan')):.3f}")

    actual_slice = slice_steps if slice_steps is not None else max(2 * spec.ndim, 8)
    save_nested_results(
        result, spec, outdir,
        nlive=nlive, slice_steps=actual_slice,
    )
    print(f"Saved posterior.npy, chain.h5, best_theta.npy, evidence.json "
          f"to {outdir}")

    if os.path.exists(pid_file):
        os.remove(pid_file)
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
