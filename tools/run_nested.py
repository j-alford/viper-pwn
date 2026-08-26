#!/usr/bin/env python3
"""Unified nested-sampling runner for all sources.

Reads ``run_config.json`` from the output directory, selects the
correct source config based on the ``source`` key (default: ``cta1``),
and runs UltraNest.

Usage
-----
    nohup python3 tools/run_nested.py results/sgr0501_run02 \\
        > results/sgr0501_run02/ns.log 2>&1 &

    # Re-run the same dir to resume (run_config "resume" key).

run_config.json keys (additive on top of the MCMC schema)
---------------------------------------------------------
    "sampler"      : optional, must be "ultranest" if set.
    "nlive"        : int, default 800 (UltraNest min_num_live_points).
    "dlogz"        : float, default 0.5 (convergence target).
    "slice_steps"  : int or null, default null -> max(2*ndim, 8).
    "resume"       : "resume" (default) or "overwrite".
    "n_cores"      : int, default 1; >1 enables Pool; overrides --ncores.
    "numba_threads": int, default cpu_count() // n_cores.
    "use_cmb_lookup": bool, default True; cached CMB IC table when compatible.

Output layout (results/<run>/)
------------------------------
    run_config.json     user-provided
    ultranest/          UltraNest's native log_dir (resume support)
    posterior.npy       equal-weighted samples (fit space)
    chain.h5            emcee.HDFBackend-readable for monitor.py
    evidence.json       summary (logz, logzerr, ncall, etc.)
    best_theta.npy      max-logL fit-space point
    ns.pid              process id (parallel to mcmc.pid)
    ns.log              stdout/stderr (when launched via nohup)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..'))   # repo root: viper, examples.<source>_config
sys.path.insert(0, _HERE)                       # tools/: run_mcmc


def _early_set_numba_threads():
    """Set NUMBA_NUM_THREADS from run_config.json BEFORE viper imports."""
    try:
        for a in sys.argv[1:]:
            if not a.startswith('-'):
                cfg_path = os.path.join(a, 'run_config.json')
                if os.path.exists(cfg_path):
                    with open(cfg_path) as f:
                        cfg = json.load(f)
                    n_threads = cfg.get('numba_threads')
                    if n_threads is None:
                        n_cores = cfg.get('n_cores', 1)
                        n_threads = max(1, (os.cpu_count() or 16) // max(int(n_cores), 1))
                    os.environ.setdefault('NUMBA_NUM_THREADS', str(int(n_threads)))
                    return
                break
    except Exception:
        pass
    os.environ.setdefault('NUMBA_NUM_THREADS', '1')


_early_set_numba_threads()

from run_mcmc import _build_from_config, validate_run_config  # noqa: E402
from viper.nested import run_nested, save_nested_results  # noqa: E402


def _check_overwrite(outdir, overwrite):
    """Refuse to start if evidence.json exists and overwrite not set."""
    ev = os.path.join(outdir, 'evidence.json')
    if os.path.exists(ev) and not overwrite:
        print(f"evidence.json already present in {outdir}.")
        print("Pass --overwrite to discard and restart, or set "
              "'resume': 'resume' in run_config.json to continue an "
              "interrupted UltraNest run (the default).")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description='PWN model nested-sampling runner')
    parser.add_argument('outdir', help='Output directory (e.g. results/cta1_run40)')
    parser.add_argument('--ncores', type=int, default=1,
                        help='Number of worker processes for vectorized '
                             'likelihood (Pool size).  1 = serial. '
                             'Overridden by run_config.json "n_cores" key.')
    parser.add_argument('--overwrite', action='store_true',
                        help='Allow overwriting an existing evidence.json.')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    config_file = os.path.join(args.outdir, 'run_config.json')
    if not os.path.exists(config_file):
        print(f"No run_config.json in {args.outdir}. Create one first.")
        return 2

    if not _check_overwrite(args.outdir, args.overwrite):
        return 1

    with open(config_file) as f:
        run_cfg = json.load(f)

    try:
        validate_run_config(run_cfg)
    except ValueError as e:
        print(f"run_config.json invalid: {e}")
        return 2

    sampler_name = run_cfg.get('sampler', 'ultranest')
    if sampler_name != 'ultranest':
        print(f"Unsupported sampler '{sampler_name}'. Only 'ultranest' "
              "is implemented; remove the 'sampler' key from "
              "run_config.json or set it to 'ultranest'.")
        return 2

    nlive = int(run_cfg.get('nlive', 800))
    dlogz = float(run_cfg.get('dlogz', 0.5))
    slice_steps = run_cfg.get('slice_steps', None)
    if slice_steps is not None:
        slice_steps = int(slice_steps)
    resume = run_cfg.get('resume', 'resume')
    cfg_n_cores = run_cfg.get('n_cores', None)
    n_cores = int(cfg_n_cores) if cfg_n_cores is not None else int(args.ncores)
    use_cmb_lookup = bool(run_cfg.get('use_cmb_lookup', True))

    pid_file = os.path.join(args.outdir, 'ns.pid')
    with open(pid_file, 'w') as f:
        f.write(str(os.getpid()))
    print(f"PID {os.getpid()} written to {pid_file}")

    log_dir = os.path.join(args.outdir, 'ultranest')

    model, obs, _spec_unused = _build_from_config(run_cfg)
    model.use_cmb_lookup = use_cmb_lookup
    print(f"use_cmb_lookup={model.use_cmb_lookup} "
          f"(default True for NS; ~1.2x speedup at threads=8 on CMB-only runs)")

    print(f"Starting NS: nlive={nlive}, dlogz={dlogz}, "
          f"slice_steps={slice_steps}, n_cores={n_cores}, resume={resume}, "
          f"log_dir={log_dir}")
    t0 = time.time()
    result, spec = run_nested(
        model, obs,
        sampler='ultranest',
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
        result, spec, args.outdir,
        nlive=nlive, slice_steps=actual_slice,
    )
    print(f"Saved posterior.npy, chain.h5, best_theta.npy, evidence.json "
          f"to {args.outdir}")

    if os.path.exists(pid_file):
        os.remove(pid_file)
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
