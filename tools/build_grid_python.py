"""Run the model at every point in the regression-grid manifest.

Reads the committed tests/reference/grid/grid.json manifest
and runs evolve_pwn(..., cpp_compat=True) at each point. Saves the
final dynamical state, electron spectrum, and photon spectrum to
tests/reference/grid/python/<name>.npz.

Usage (from repo root):
    VIPER_CONSTANTS=gsz2009 python3 tools/build_grid_python.py
    VIPER_CONSTANTS=gsz2009 python3 tools/build_grid_python.py \
        --names g54_baseline cta1_aliu_pre
    VIPER_CONSTANTS=gsz2009 python3 tools/build_grid_python.py \
        --skip-existing
"""
from __future__ import annotations
import argparse
import datetime
import json
import os
import platform
import subprocess
import sys
import time
import traceback

if os.environ.get('VIPER_CONSTANTS') != 'gsz2009':
    sys.stderr.write(
        'ERROR: VIPER_CONSTANTS must be set to "gsz2009" in the '
        'environment BEFORE invoking this script. Re-run as:\n'
        '  VIPER_CONSTANTS=gsz2009 python3 tools/build_grid_python.py\n'
    )
    sys.exit(2)

import numpy as np

try:
    import numba
    NUMBA_VERSION = numba.__version__
except ImportError:
    NUMBA_VERSION = None

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..'))
GRID = os.path.join(REPO, 'tests', 'reference', 'grid')
OUT  = os.path.join(GRID, 'python')
RUNS_PATH   = os.path.join(GRID, 'grid_python_runs.jsonl')
STATUS_PATH = os.path.join(GRID, 'grid_python_status.json')

sys.path.insert(0, REPO)
from viper import constants as C
from viper.scalar_params import PWNScalarParams
from viper.evolution import evolve_pwn


def _run_context() -> dict:
    """Snapshot env metadata once per build invocation."""
    try:
        sha = subprocess.check_output(
            ['git', '-C', REPO, 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        sha = None
    now = datetime.datetime.now(datetime.timezone.utc)
    return {
        'timestamp': now.replace(tzinfo=None).isoformat(timespec='seconds') + 'Z',
        'git_sha': sha,
        'python_version': platform.python_version(),
        'numba_version': NUMBA_VERSION,
        'constants': os.environ.get('VIPER_CONSTANTS', '<unset>'),
    }


def _append_run(name: str, status: str, elapsed_s: float, category: str,
                ctx: dict) -> None:
    """Append one JSONL record.  Always 'a' mode; never truncate."""
    record = {
        'name': name, 'status': status, 'elapsed_s': elapsed_s,
        'category': category, **ctx,
    }
    with open(RUNS_PATH, 'a') as f:
        f.write(json.dumps(record) + '\n')


def _refresh_status_from_runs() -> None:
    """Rebuild STATUS_PATH from the full RUNS_PATH log."""
    if not os.path.exists(RUNS_PATH):
        return
    latest: dict[str, dict] = {}
    first_seen: dict[str, int] = {}
    with open(RUNS_PATH) as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            name = rec['name']
            if name not in first_seen:
                first_seen[name] = idx
            latest[name] = {
                'name': name,
                'status': rec['status'],
                'elapsed_s': rec['elapsed_s'],
                'category': rec.get('category', 'unknown'),
            }
    sys.path.insert(0, HERE)
    try:
        from build_grid import ANCHORS, lhs_points, axis_points
    except ImportError:
        pass  # build_grid may be absent
    else:
        valid_names = (
            {p.name for p in ANCHORS}
            | {p.name for p in lhs_points()}
            | {p.name for p in axis_points()}
        )
        latest = {n: r for n, r in latest.items() if n in valid_names}
    records = sorted(latest.values(), key=lambda r: first_seen[r['name']])
    with open(STATUS_PATH, 'w') as f:
        json.dump(records, f, indent=2)

# Final-state fields recorded per grid point.
DYN_FIELDS = [
    'time', 'rad_pwn', 'rad_snr', 'rad_rs', 'rad_psr',
    'bpwn', 'ad_lum_40', 'synch_lum_40', 'ic_lum_40',
    'edot_40', 'epwn_50', 'epwnb_50', 'epwnp_50',
    'vpwn', 'vejpwn', 'mswpwn', 'ppwn', 'psnrpwn',
    'rhoejpwn', 'vsnr', 'eparfudge', 'delta_t',
]


def run_one(point: dict, force: bool = False) -> dict:
    out_path = os.path.join(OUT, f'{point["name"]}.npz')
    if not force and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return {'name': point['name'], 'status': 'skip',
                'reason': 'output exists', 'elapsed_s': 0.0}

    # Build the params
    params = PWNScalarParams(
        esn=point['esn'], mej=point['mej'], n_ism=point['n_ism'],
        braking_index=point['braking_index'], tau_sd=point['tau_sd'],
        period=point['period'], p_dot=point['p_dot'],
        eta_b=point['eta_b'],
        e_min=point['e_min'], e_max=point['e_max'],
        e_break=point['e_break'], p1=point['p1'], p2=point['p2'],
        f_max=point.get('f_max', 0.0), kT_max=point.get('kT_max', 0.0),
        eta_g=point.get('eta_g', 0.0), vel_psr=point.get('vel_psr', 0.0),
        n_ic=len(point.get('ic_temps', [])),
        ic_temps=list(point.get('ic_temps', [])),
        ic_norms=list(point.get('ic_norms', [])),
    )

    t0 = time.time()
    try:
        result = evolve_pwn(
            cpp_compat=True,
            esn_51=params.esn, mej=params.mej, n_ism=params.n_ism,
            braking_index=params.braking_index, tau_sd=params.tau_sd,
            age=params.true_age, e0_37=params.e0_37,
            vel_psr=params.vel_psr, eta_g=params.eta_g, eta_b=params.eta_b,
            e_min=params.e_min, e_max=params.e_max,
            e_break=params.e_break, p1=params.p1, p2=params.p2,
            f_max=params.f_max, kT_max=params.kT_max,
            n_ic=params.n_ic,
            ic_temps=params.ic_temps, ic_norms=params.ic_norms,
            dyn_step=-1, elec_step=-1, phot_step=-1,
        )
    except Exception as exc:
        elapsed = time.time() - t0
        return {'name': point['name'], 'status': 'crash',
                'reason': f'{type(exc).__name__}: {exc}',
                'traceback': traceback.format_exc()[-1000:],
                'elapsed_s': elapsed}
    elapsed = time.time() - t0

    # Pack the final-state dyninfo and the spectra.
    dyn = result.dyninfo[-1]
    dyn_dict = {f: float(getattr(dyn, f)) for f in DYN_FIELDS}

    # total target-field energy density (eV/cm^3); downstream preflight check
    u_cmb = C.radiation_const * C.T_CMB ** 4 * C.eV_per_erg
    uic = float(u_cmb + sum(params.ic_norms))

    np.savez(
        out_path,
        **{f'dyn_{k}': v for k, v in dyn_dict.items()},
        elec_bin_avg=result.elec_grid.bin_avg,
        elec_spectrum=result.electron_spectrum,
        phot_freq_avg=result.phot_grid.bin_avg,
        phot_spectrum=result.photon_spectrum,
        uic=uic,
    )

    return {'name': point['name'], 'status': 'ok',
            'elapsed_s': elapsed}


def main():
    ap = argparse.ArgumentParser(description='Build Python-port grid.')
    ap.add_argument('--names', nargs='+',
                    help='Only run the named points')
    ap.add_argument('--skip-existing', action='store_true',
                    help='Skip points whose .npz already exists')
    ap.add_argument('--force', action='store_true',
                    help='Overwrite existing outputs')
    args = ap.parse_args()

    manifest_path = os.path.join(GRID, 'grid.json')
    if not os.path.exists(manifest_path):
        sys.stderr.write(
            f'ERROR: grid manifest not found: {manifest_path}\n'
            f'The committed manifest is missing; restore it from git.\n'
        )
        sys.exit(1)

    manifest = json.load(open(manifest_path))
    units = manifest.get('ic_norm_units')
    if units != 'eV_per_cm3':
        sys.stderr.write(
            f'ERROR: grid manifest ic_norm_units={units!r}, expected '
            f'"eV_per_cm3" (ic_norms are energy densities in eV/cm^3).\n'
        )
        sys.exit(3)
    points = manifest['points']
    if args.names:
        wanted = set(args.names)
        points = [p for p in points if p['name'] in wanted]
        missing = wanted - {p['name'] for p in points}
        if missing:
            print(f'WARNING: unknown names: {sorted(missing)}',
                  file=sys.stderr)

    os.makedirs(OUT, exist_ok=True)
    print(f'Total grid points to attempt: {len(points)}')

    ctx = _run_context()
    results = []
    for i, point in enumerate(points, 1):
        tag = f'[{i:3d}/{len(points):3d}]'
        print(f'{tag} {point["category"]:8s} {point["name"]:35s}',
              end=' ', flush=True)
        res = run_one(point, force=args.force and not args.skip_existing)
        results.append({**res, 'category': point['category']})
        _append_run(
            name=res['name'], status=res['status'],
            elapsed_s=res['elapsed_s'], category=point['category'],
            ctx=ctx,
        )
        print(f'-> {res["status"]:10s} {res["elapsed_s"]:6.2f}s',
              flush=True)

    _refresh_status_from_runs()

    ok   = sum(1 for r in results if r['status'] == 'ok')
    skip = sum(1 for r in results if r['status'] == 'skip')
    bad  = len(results) - ok - skip
    total_t = sum(r['elapsed_s'] for r in results)
    print()
    print(f'Summary: {ok} ok, {skip} skipped, {bad} bad. '
          f'Total wall time: {total_t/60:.2f} min')
    if bad:
        print('Failures:')
        for r in results:
            if r['status'] not in ('ok', 'skip'):
                print(f'  {r["name"]:35s} {r["status"]:10s} '
                      f'{str(r.get("reason",""))[:120]!r}')


if __name__ == '__main__':
    main()
