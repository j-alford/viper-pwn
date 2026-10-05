#!/usr/bin/env python3
"""Monitor an MCMC chain (running or finished) via its HDF5 backend.

Usage:
    python3 tools/monitor.py results/g54_run02/chain.h5
    python3 tools/monitor.py results/g54_run02/chain.h5 --ref results/g54_run01/best_theta.npy
"""
import argparse
import sys
import os
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..'))   # repo root: viper

from viper.fitting import (
    predict_observables, log_probability, log_prior,
    _theta_to_evolve_args,
)
from viper.source import (
    SourceError, build_from_source, load_run, source_settings,
)


# ANSI colors
G = '\033[32m'
R = '\033[31m'
Y = '\033[33m'
B = '\033[1m'
W = '\033[0m'


def main():
    parser = argparse.ArgumentParser(description='Monitor an MCMC chain')
    parser.add_argument('h5file', help='Path to chain.h5')
    parser.add_argument('--ref', help='Reference best_theta.npy for comparison',
                        default=None)
    parser.add_argument('--plain', action='store_true',
                        help='Plain output (no ANSI colors, for Slack/email)')
    parser.add_argument('--theta', default=None,
                        help='Path to .npy theta to display instead of chain best')
    parser.add_argument('--save-best', action='store_true',
                        help="Write the chain's best theta to <run>/best_theta.npy "
                             '(the monitor is read-only by default)')
    args = parser.parse_args()

    global G, R, Y, B, W
    if args.plain:
        G = R = Y = B = W = ''

    import emcee
    backend = emcee.backends.HDFBackend(args.h5file, read_only=True)

    nsteps = backend.iteration
    nsteps_total = None
    if nsteps == 0:
        print("No steps recorded yet.")
        return

    h5_mtime = os.path.getmtime(args.h5file)
    h5_ctime = os.path.getctime(args.h5file)

    lps = backend.get_log_prob()
    best_idx = np.unravel_index(np.argmax(lps), lps.shape)
    chain_best_theta = backend.get_chain()[best_idx]
    best_lp = lps[best_idx]
    acc = np.mean(backend.accepted / backend.iteration)

    run_dir = os.path.dirname(os.path.abspath(args.h5file))
    try:
        src = load_run(run_dir)
        model, obs, spec = build_from_source(src)
    except SourceError as e:
        print(e)
        return

    # extended and nested-sampling chains run past the planned length
    if backend.iteration <= source_settings(src)['FIT']['nsteps']:
        nsteps_total = source_settings(src)['FIT']['nsteps']

    if args.theta and os.path.exists(args.theta):
        custom_theta = np.load(args.theta)
        if len(custom_theta) == spec.ndim:
            best_theta = custom_theta
            print(f"Using theta from: {args.theta}")
        else:
            print(f"{Y}--theta has {len(custom_theta)} params, spec has {spec.ndim} -- ignoring{W}")
            best_theta = chain_best_theta
    else:
        best_theta = chain_best_theta

    # Verify log_prob matches
    lp_check = log_probability(best_theta, spec)
    # posterior log_prob; chi2 excludes prior penalty
    lp_prior = log_prior(best_theta, spec)
    if not np.isfinite(lp_prior):
        lp_prior = 0.0
    _pr = f", prior dchi2 = {-2*lp_prior:.2f}" if lp_prior != 0.0 else ""

    print(f"\n{B}MCMC Monitor{W}")
    print(f"File: {args.h5file}")
    progress = f"Steps: {nsteps}"
    if nsteps_total:
        progress += f"/{nsteps_total} ({100*nsteps/nsteps_total:.0f}%)"
    print(f"{progress}  Acceptance: {acc:.3f}")
    custom_lp = log_probability(best_theta, spec)
    if args.theta and os.path.exists(args.theta):
        print(f"Best log_prob: {custom_lp:.4f}  "
              f"(chi2 = {-2*(custom_lp - lp_prior):.2f}{_pr})  [custom theta]")
    else:
        print(f"Best log_prob: {best_lp:.4f}  "
              f"(chi2 = {-2*(best_lp - lp_prior):.2f}{_pr})")
    if abs(lp_check - best_lp) > 0.1:
        print(f"{R}WARNING: recomputed log_prob = {lp_check:.4f} "
              f"(chi2 = {-2*(lp_check - lp_prior):.2f}{_pr}) -- spec mismatch!{W}")

    import time
    elapsed = h5_mtime - h5_ctime
    if elapsed > 0 and nsteps > 1:
        sec_per_step = elapsed / nsteps
        if nsteps_total and nsteps < nsteps_total:
            remaining = sec_per_step * (nsteps_total - nsteps)
            hrs = remaining / 3600
            if hrs >= 1:
                print(f"  Pace: {sec_per_step:.1f} s/step  "
                      f"ETA: ~{hrs:.1f} hrs remaining")
            else:
                print(f"  Pace: {sec_per_step:.1f} s/step  "
                      f"ETA: ~{remaining/60:.0f} min remaining")
        else:
            print(f"  Pace: {sec_per_step:.1f} s/step")

    # Evaluate best theta
    pred_now, result_now = predict_observables(best_theta, spec)
    dyn = result_now.dyninfo[-1]

    # Load reference if provided
    has_ref = False
    if args.ref and os.path.exists(args.ref):
        ref_theta = np.load(args.ref)
        if len(ref_theta) == spec.ndim:
            pred_ref, _ = predict_observables(ref_theta, spec)
            has_ref = True
        else:
            print(f"{Y}Reference has {len(ref_theta)} params, "
                  f"spec has {spec.ndim} -- skipping comparison{W}")

    # Walker diagnostics
    chain = backend.get_chain()  # (nsteps, nwalkers, ndim)
    nsteps_done, nwalkers_total, ndim = chain.shape
    acc_per_walker = backend.accepted / nsteps_done

    print(f"\n{B}Walker diagnostics ({nwalkers_total} walkers, {nsteps_done} steps){W}")

    stuck = int(np.sum(acc_per_walker < 0.05))
    slow = int(np.sum((acc_per_walker >= 0.05) & (acc_per_walker < 0.15)))
    good = int(np.sum((acc_per_walker >= 0.15) & (acc_per_walker < 0.40)))
    fast = int(np.sum(acc_per_walker >= 0.40))
    c_stuck = R if stuck > 0 else W
    c_slow = Y if slow > nwalkers_total * 0.25 else W
    c_good = G if good > nwalkers_total * 0.5 else W
    c_fast = G if fast > 0 else W
    print(f"  Acceptance:  {c_stuck}stuck(<5%)={stuck}{W}  "
          f"{c_slow}slow(5-15%)={slow}{W}  "
          f"{c_good}good(15-40%)={good}{W}  "
          f"{c_fast}fast(>40%)={fast}{W}")
    med_acc = np.median(acc_per_walker)
    c_med = G if med_acc >= 0.15 else (Y if med_acc >= 0.05 else R)
    print(f"  Range: {acc_per_walker.min():.3f} - {acc_per_walker.max():.3f}  "
          f"median {c_med}{med_acc:.3f}{W}")

    lps_now = lps[-1, :]
    lps_best_now = np.max(lps_now)
    within_2 = int(np.sum(lps_now > lps_best_now - 1))
    within_5 = int(np.sum(lps_now > lps_best_now - 2.5))
    within_10 = int(np.sum(lps_now > lps_best_now - 5))
    spread = -2*np.min(lps_now) - (-2*lps_best_now)
    c_spread = G if spread < 10 else (Y if spread < 30 else R)
    c_w2 = G if within_2 > nwalkers_total * 0.3 else (Y if within_2 > 5 else R)
    # posterior, hence -2logP not chi2
    print(f"  -2logP at step {nsteps_done}: best={-2*lps_best_now:.1f}  "
          f"worst={-2*np.min(lps_now):.1f}  "
          f"spread={c_spread}{spread:.1f}{W}")
    print(f"  Walkers near best:  "
          f"{c_w2}d(-2logP)<2: {within_2}{W}  "
          f"d(-2logP)<5: {within_5}  d(-2logP)<10: {within_10}")

    try:
        tau = backend.get_autocorr_time(quiet=True)
        tau_max = np.max(tau)
        n_indep = nsteps_done / tau_max
        c_tau = G if n_indep > 10 else (Y if n_indep > 3 else R)
        print(f"  Autocorr: max tau={tau_max:.0f} steps  "
              f"{c_tau}~{n_indep:.0f} independent samples{W}")
    except Exception:
        print(f"  Autocorr: {Y}not enough steps yet{W}")

    pos_now = chain[-1, :, :]
    print(f"\n{B}{'Parameter':<18} {'walker std':>10} {'moved':>10} {'std/moved':>10}{W}")
    print("-" * 50)
    for i, fp in enumerate(spec.free_params):
        wstd = np.std(pos_now[:, i])
        pnow_val = best_theta[i]
        if has_ref:
            moved = abs(pnow_val - ref_theta[i])
        else:
            moved = abs(pnow_val - np.mean(chain[0, :, i]))
        if moved > 1e-10:
            ratio = wstd / moved
            c_r = G if 0.3 < ratio < 3.0 else (Y if 0.1 < ratio < 10 else R)
            ratio_str = f"{c_r}{ratio:.2f}{W}"
        else:
            ratio_str = f"{Y}---{W}"
        print(f"  {fp.name:<16} {wstd:>10.4f} {moved:>10.4f} {ratio_str:>20}")

    # Parameters
    print(f"\n{B}{'Parameter':<16} {'delta':>8} {'value':>10}  {'':7s}  {'bounds':<20s}{W}")
    print("-" * 80)

    _PARAM_UNITS = {
        'esn': 'e51 erg', 'mej': 'Msun', 'n_ism': 'cm-3',
        'braking_index': '', 'tau_sd': 'yr', 'eta_b': '',
        'e_min': 'GeV', 'e_max': 'TeV', 'e_break': 'GeV',
        'p1': '', 'p2': '', 'distance': 'kpc',
        'ic_temp_1': 'K', 'ic_norm_1': 'scale',
        'ic_temp_2': 'K', 'ic_norm_2': 'scale',
    }

    _PARAM_SCALE = {
        'distance': 1e-3,
        'e_max': 1e-3,
    }

    total_prior = 0.0
    for i, fp in enumerate(spec.free_params):
        vnow = best_theta[i]
        pnow = 10**vnow if fp.log else vnow
        scale = _PARAM_SCALE.get(fp.name, 1.0)
        pdisp = pnow * scale
        line = f"{fp.name:<16}"

        if has_ref:
            vref = ref_theta[i]
        else:
            vref = np.mean(chain[0, :, i])
        pref = 10**vref if fp.log else vref
        ratio = pnow / pref if pref != 0 else 1.0
        arrow = "↑" if ratio > 1.001 else ("↓" if ratio < 0.999 else " ")
        diff = abs(vnow - vref)
        c = G if diff > 0.01 else W
        unit = _PARAM_UNITS.get(fp.name, '')
        line += f" {c}{arrow}{ratio:.3f}x{W} {pdisp:>10.4g}"
        if unit:
            line += f" {unit:<7s}"
        else:
            line += f" {'':7s}"

        lo, hi = fp.bounds
        if fp.log:
            lo_p, hi_p = 10**lo * scale, 10**hi * scale
        else:
            lo_p, hi_p = lo * scale, hi * scale
        line += f"  [{lo_p:.3g}, {hi_p:.3g}]"

        if fp.prior is not None:
            center, sigma = fp.prior
            if getattr(fp, 'prior_log', False):
                tnow = ((vnow - center) / sigma)**2
            else:
                tnow = ((pnow - center) / sigma)**2
            total_prior += tnow
            c_pr = G if tnow < 0.5 else (Y if tnow < 2.0 else R)
            line += f"  {c_pr}prior({center},{sigma}) chi2={tnow:.2f}{W}"

        print(line)

    # Frozen parameters
    free_names = {fp.name for fp in spec.free_params}
    known_params = set(_PARAM_UNITS.keys())
    skip = {'period', 'pdot', 'edot', 'tau_char', 'eta_g', 'vel_psr',
            'f_max', 'kT_max', 'p_dot'}
    for p in model.parameters:
        if not p.frozen or p.name in skip or p.name in free_names:
            continue
        if p.name not in known_params:
            continue
        scale = _PARAM_SCALE.get(p.name, 1.0)
        unit = _PARAM_UNITS.get(p.name, '')
        val = p.value
        if hasattr(val, 'value'):
            val = float(val.value)
        pdisp = val * scale
        line = f"{p.name:<16}  {'':8s}  {pdisp:>10.4g}"
        if unit:
            line += f" {unit:<7s}"
        else:
            line += f" {'':7s}"
        line += f"  [frozen]"
        print(line)

    print(f"\nDynamics: R_pwn={dyn.rad_pwn:.2f} pc, "
          f"R_snr={dyn.rad_snr:.2f} pc, B={dyn.bpwn:.3f} uG")
    print(f"  Age = {dyn.time:.0f} yr ({dyn.time/1000:.1f} kyr), "
          f"v_pwn = {dyn.vpwn:.0f} km/s")

    try:
        from viper.evolution import evolve_pwn
        kwargs = _theta_to_evolve_args(best_theta, spec)
        kwargs['dyn_step'] = 1
        kwargs['elec_step'] = -1
        kwargs['phot_step'] = -1
        full_result = evolve_pwn(**kwargs)
        rs_hit = False
        rs_time = None
        for di in full_result.dyninfo:
            if di.rad_rs > 0 and di.rad_rs <= di.rad_pwn:
                rs_hit = True
                rs_time = di.time
                break
        if rs_hit:
            c_rs = Y
            print(f"  {c_rs}Reverse shock hit PWN at t = {rs_time:.0f} yr "
                  f"({rs_time/1000:.1f} kyr){W}")
        else:
            print(f"  {G}Reverse shock has not reached PWN{W}")
    except Exception as e:
        print(f"  {Y}Could not check reverse shock: {e}{W}")

    # Residuals
    print(f"\n{B}{'Observable':<30} {'data':>12} {'model':>10} {'pull':>7} {'wt':>4} {'wt*chi2':>8}{W}")
    print("-" * 75)

    def _pull_color(dsnow):
        a = abs(dsnow)
        if a < 0.5:
            return G
        if a > 1.5:
            return R
        return W

    total_wchi2 = 0.0
    total_chi2 = 0.0

    for k, ob in enumerate(spec.observables):
        model_val = pred_now[k]

        if ob.upper_limit:
            if model_val <= ob.value:
                snow, dsnow = "OK", 0.0
            else:
                dsnow = (model_val - ob.value) / ob.limit_sigma
                snow = f"{dsnow:+.2f}"
            wchi2 = ob.weight * dsnow**2
            total_wchi2 += wchi2
            total_chi2 += dsnow**2
            target_str = f"<{ob.value:.3g}"
            c = _pull_color(dsnow)
            print(f"{ob.label:<30} {target_str:>12} {model_val:>10.3g} {c}{snow:>7}{W} "
                  f"{ob.weight:>4.1f} {wchi2:>8.2f}")
            continue

        if getattr(ob, 'lower_limit', False):
            if model_val >= ob.value:
                snow, dsnow = "OK", 0.0
            else:
                dsnow = (model_val - ob.value) / ob.limit_sigma
                snow = f"{dsnow:+.2f}"
            wchi2 = ob.weight * dsnow**2
            total_wchi2 += wchi2
            total_chi2 += dsnow**2
            target_str = f">{ob.value:.3g}"
            c = _pull_color(dsnow)
            print(f"{ob.label:<30} {target_str:>12} {model_val:>10.3g} {c}{snow:>7}{W} "
                  f"{ob.weight:>4.1f} {wchi2:>8.2f}")
            continue

        if ob.range_lo is not None and ob.range_hi is not None:
            def _range_dsig(pred, lo, hi, err):
                if lo <= pred <= hi:
                    return 0.0
                elif pred < lo:
                    return (lo - pred) / err
                else:
                    return (hi - pred) / err
            dsnow = _range_dsig(model_val, ob.range_lo, ob.range_hi, ob.error)
            target_str = f"[{ob.range_lo:.1f},{ob.range_hi:.1f}]"
        else:
            dsnow = (ob.value - model_val) / ob.error
            target_str = f"{ob.value:.3g}"

        wchi2 = ob.weight * dsnow**2
        chi2_term = dsnow**2
        total_wchi2 += wchi2
        total_chi2 += chi2_term

        c = _pull_color(dsnow)
        print(f"{ob.label:<30} {target_str:>12} {model_val:>10.3g} "
              f"{c}{dsnow:>+7.2f}{W} {ob.weight:>4.1f} {wchi2:>8.2f}")

    ndof = len(spec.observables) - len(spec.free_params)
    red_chi2 = total_chi2 / ndof if ndof > 0 else 0.0
    c_red = G if red_chi2 < 2.0 else (Y if red_chi2 < 3.0 else R)
    print(f"\n  {B}Unweighted obs chi2 = {total_chi2:.2f}{W}")
    print(f"  {c_red}{B}Reduced chi2        = {total_chi2:.2f}/{ndof} = {red_chi2:.2f}{W}")
    print(f"  {B}Weighted obs chi2   = {total_wchi2:.2f}{W}")
    print(f"  {B}Prior chi2          = {total_prior:.2f}{W}")
    print(f"  {B}Total (-2*log_prob) = {total_wchi2 + total_prior:.2f}{W}")
    print(f"  {B}Backend -2*log_prob = {-2*best_lp:.2f}{W}")

    # writes chain best, never --theta
    if args.save_best:
        best_path = os.path.join(run_dir, 'best_theta.npy')
        np.save(best_path, chain_best_theta)
        print(f"\n  Saved {best_path} (chain best)")


if __name__ == '__main__':
    main()

