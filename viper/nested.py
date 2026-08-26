"""
Nested-sampling fits for the PWN model.

Wraps UltraNest (https://johannesbuchner.github.io/UltraNest/) so any
``PWNModel`` + ``Observable`` list configured for ``run_mcmc`` can be
explored with NS.

Warning: log-likelihood is -chi^2/2 without normalization and weights
are non-unity; logz is NOT meaningful for model comparison, posteriors are.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# Prior transform

def prior_transform(u, spec):
    """Map unit-cube vector ``u`` in [0,1]^ndim to fit-space theta vector.

    Mirrors ``viper.fitting.log_prior``: no prior -> uniform on bounds;
    ``prior_log=True`` or ``log=False`` -> truncated Gaussian in fit
    space; prior on a log-sampled parameter -> truncated Gaussian in
    physical space, then log10.
    Measure difference: the last branch draws in physical space;
    emcee's ``log_prior`` acts on the flat-in-theta measure.
    """
    from scipy.stats import truncnorm

    u = np.asarray(u, dtype=float)
    theta = np.empty_like(u, dtype=float)
    for i, fp in enumerate(spec.free_params):
        lo_fit, hi_fit = fp.bounds
        ui = u[i]
        if fp.prior is None:
            theta[i] = lo_fit + ui * (hi_fit - lo_fit)
        elif fp.prior_log or not fp.log:
            mu, sigma = fp.prior
            a = (lo_fit - mu) / sigma
            b = (hi_fit - mu) / sigma
            theta[i] = truncnorm.ppf(ui, a, b, loc=mu, scale=sigma)
        else:
            mu, sigma = fp.prior
            lo_phys = 10.0 ** lo_fit
            hi_phys = 10.0 ** hi_fit
            a = (lo_phys - mu) / sigma
            b = (hi_phys - mu) / sigma
            phys_val = truncnorm.ppf(ui, a, b, loc=mu, scale=sigma)
            phys_val = max(phys_val, lo_phys)
            theta[i] = np.log10(phys_val)
    return theta


def prior_transform_vec(U, spec):
    """Vectorized prior transform: (K, ndim) -> (K, ndim)."""
    U = np.asarray(U, dtype=float)
    out = np.empty_like(U)
    for k in range(U.shape[0]):
        out[k] = prior_transform(U[k], spec)
    return out


# Runner

def run_nested(model, observables, *,
               sampler='ultranest', integration='pc_full',
               nlive=800, dlogz=0.5, slice_steps=None, n_cores=1,
               log_dir=None, resume='resume',
               show_status=True, **sampler_kwargs):
    """Run a nested-sampling fit on a PWNModel.

    Parameters
    ----------
    model : PWNModel
    observables : list of Observable
    sampler : {'ultranest'}
        Backend to use.  Currently only UltraNest.
    integration : str
        Forwarded to :func:`viper.fitting._build_fit_spec`.
    nlive : int
        Minimum number of live points.
    dlogz : float
        Convergence target on the log-evidence remainder.
    slice_steps : int, optional
        Slice-sampler nsteps.  Defaults to ``max(2*ndim, 8)``.
    n_cores : int
        If ``> 1``, evaluate the likelihood in parallel via a
        ``multiprocessing.Pool`` (vectorized mode).  Default 1 (serial).
    log_dir : str, optional
        UltraNest log directory (used for incremental save / resume).
    resume : {'resume', 'overwrite', 'subfolder'}
        Forwarded to UltraNest.
    show_status : bool
        Print progress.
    **sampler_kwargs
        Additional kwargs forwarded to ``ReactiveNestedSampler.run``.

    Returns
    -------
    result : dict
        UltraNest's result dict (samples, logz, logzerr, ncall, ...).
    spec : _FitSpec
        Parallel to ``run_mcmc``'s second return value.
    """
    if sampler == 'ultranest':
        return _run_ultranest(
            model, observables, integration=integration,
            nlive=nlive, dlogz=dlogz, slice_steps=slice_steps,
            n_cores=n_cores,
            log_dir=log_dir, resume=resume, show_status=show_status,
            **sampler_kwargs,
        )
    raise ValueError(
        f"Unknown sampler '{sampler}'. Currently supported: 'ultranest'. "
        "dynesty/nautilus would be added as additional _run_<name> branches."
    )


def _run_ultranest(model, observables, *,
                   integration, nlive, dlogz, slice_steps, n_cores,
                   log_dir, resume, show_status, **extra):
    from .fitting import _build_fit_spec

    model.validate()
    spec = _build_fit_spec(model, observables, integration=integration)

    if slice_steps is None:
        slice_steps = max(2 * spec.ndim, 8)

    logger.info(
        "Starting NS: nlive=%d, dlogz=%.3g, slice_steps=%d, ndim=%d, "
        "n_cores=%d, log_dir=%s",
        nlive, dlogz, slice_steps, spec.ndim, n_cores, log_dir,
    )

    if n_cores == 1:
        return _run_ultranest_serial(
            spec, nlive=nlive, dlogz=dlogz, slice_steps=slice_steps,
            log_dir=log_dir, resume=resume, show_status=show_status,
            **extra,
        )
    return _run_ultranest_parallel(
        spec, n_cores=n_cores, nlive=nlive, dlogz=dlogz,
        slice_steps=slice_steps, log_dir=log_dir, resume=resume,
        show_status=show_status, **extra,
    )


def _run_ultranest_serial(spec, *, nlive, dlogz, slice_steps,
                           log_dir, resume, show_status, **extra):
    import ultranest
    import ultranest.stepsampler
    from .fitting import log_likelihood

    def _loglike(theta):
        ll = log_likelihood(np.asarray(theta, dtype=float), spec)
        if not np.isfinite(ll):
            return -1e300
        return ll

    def _prior(u):
        return prior_transform(np.asarray(u, dtype=float), spec)

    sampler_obj = ultranest.ReactiveNestedSampler(
        param_names=spec.param_names,
        loglike=_loglike,
        transform=_prior,
        log_dir=log_dir,
        resume=resume,
    )
    sampler_obj.stepsampler = ultranest.stepsampler.SliceSampler(
        nsteps=slice_steps,
        generate_direction=ultranest.stepsampler.generate_mixture_random_direction,
    )
    result = sampler_obj.run(
        min_num_live_points=nlive,
        dlogz=dlogz,
        show_status=show_status,
        viz_callback=None,
        **extra,
    )
    return result, spec


def _run_ultranest_parallel(spec, *, n_cores, nlive, dlogz, slice_steps,
                             log_dir, resume, show_status, **extra):
    import ultranest
    import ultranest.stepsampler
    from multiprocessing import Pool
    from .fitting import _init_worker, _log_like_worker

    def _vec_prior(u_batch):
        u_batch = np.asarray(u_batch, dtype=float)
        return np.array([prior_transform(u, spec) for u in u_batch])

    pool = Pool(n_cores, initializer=_init_worker, initargs=(spec,))
    try:
        def _vec_loglike(theta_batch):
            theta_batch = np.asarray(theta_batch, dtype=float)
            results = pool.map(_log_like_worker, list(theta_batch))
            arr = np.asarray(results, dtype=float)
            arr = np.where(np.isfinite(arr), arr, -1e300)
            return arr

        sampler_obj = ultranest.ReactiveNestedSampler(
            param_names=spec.param_names,
            loglike=_vec_loglike,
            transform=_vec_prior,
            log_dir=log_dir,
            resume=resume,
            vectorized=True,
        )
        sampler_obj.stepsampler = ultranest.stepsampler.SliceSampler(
            nsteps=slice_steps,
            generate_direction=ultranest.stepsampler.generate_mixture_random_direction,
        )
        result = sampler_obj.run(
            min_num_live_points=nlive,
            dlogz=dlogz,
            show_status=show_status,
            viz_callback=None,
            **extra,
        )
    finally:
        pool.close()
        pool.join()
    return result, spec


# Output saving + emcee-format chain adapter

def write_emcee_chain(samples, loglikes, ndim, h5path, nwalkers=None):
    """Write equal-weighted NS samples to emcee-format HDF5.

    Samples reshape to ``(nsteps, nwalkers, ndim)``; if not divisible
    by ``nwalkers``, leading samples are dropped.
    """
    import h5py
    import emcee

    if nwalkers is None:
        nwalkers = max(ndim * 4, 16)
    nsteps = len(samples) // nwalkers
    if nsteps == 0:
        raise ValueError(
            f"Too few samples ({len(samples)}) for nwalkers={nwalkers}. "
            "Reduce nwalkers or run NS longer."
        )
    n_used = nwalkers * nsteps
    chain_3d = samples[:n_used].reshape(nsteps, nwalkers, ndim)
    lp_2d = loglikes[:n_used].reshape(nsteps, nwalkers)

    if os.path.exists(h5path):
        os.remove(h5path)
    backend = emcee.backends.HDFBackend(h5path)
    backend.reset(nwalkers, ndim)
    with h5py.File(h5path, 'a') as f:
        g = f['mcmc']
        g.attrs['iteration'] = nsteps
        for k in ('chain', 'log_prob', 'accepted'):
            if k in g:
                del g[k]
        g.create_dataset('chain', data=chain_3d.astype(np.float64),
                         maxshape=(None, nwalkers, ndim))
        g.create_dataset('log_prob', data=lp_2d.astype(np.float64),
                         maxshape=(None, nwalkers))
        g.create_dataset('accepted', data=np.full(nwalkers, nsteps, dtype=int))


def save_nested_results(result, spec, outdir, *,
                         nlive=None, slice_steps=None,
                         h5_nwalkers=None):
    """Persist NS results to ``outdir`` in a monitor-compatible layout.

    Writes:
      - ``posterior.npy``  -- equal-weighted samples in fit space (N, ndim).
      - ``chain.h5``       -- emcee.HDFBackend-readable reshape of the same.
      - ``best_theta.npy`` -- max-logL fit-space point.
      - ``evidence.json``  -- run summary (logz, logzerr, ncall, etc.).

    Parameters
    ----------
    result : dict
        UltraNest's return value.
    spec : _FitSpec
        Returned alongside ``result`` by :func:`run_nested`.
    outdir : str
        Run directory (must exist).
    nlive, slice_steps : optional
        Recorded into ``evidence.json``.
    h5_nwalkers : int, optional
        Forwarded to :func:`write_emcee_chain`.
    """
    samples = np.asarray(result['samples'], dtype=float)
    if samples.ndim != 2 or samples.shape[1] != spec.ndim:
        raise ValueError(
            f"result['samples'] has unexpected shape {samples.shape}; "
            f"expected (N, {spec.ndim})."
        )

    loglikes = _extract_sample_loglikes(result, samples, spec)

    np.save(os.path.join(outdir, 'posterior.npy'), samples)

    best_idx = int(np.argmax(loglikes))
    best_theta = samples[best_idx]
    np.save(os.path.join(outdir, 'best_theta.npy'), best_theta)

    write_emcee_chain(
        samples=samples,
        loglikes=loglikes,
        ndim=spec.ndim,
        h5path=os.path.join(outdir, 'chain.h5'),
        nwalkers=h5_nwalkers,
    )

    summary = {
        'logz': float(result.get('logz', float('nan'))),
        'logzerr': float(result.get('logzerr', float('nan'))),
        'logz_max': float(np.max(loglikes)),
        'ncall': int(result.get('ncall', 0)),
        'nlive': int(nlive) if nlive is not None else None,
        'slice_steps': int(slice_steps) if slice_steps is not None else None,
        'ndim': int(spec.ndim),
        'param_names': list(spec.param_names),
        'best_theta_fit': [float(x) for x in best_theta],
        'best_logl': float(loglikes[best_idx]),
        'n_samples': int(len(samples)),
    }
    with open(os.path.join(outdir, 'evidence.json'), 'w') as f:
        json.dump(summary, f, indent=2)


def _extract_sample_loglikes(result, samples, spec):
    """Return a per-sample log-likelihood array, computing on the fly if
    UltraNest didn't supply one for the equal-weighted samples."""
    ws = result.get('weighted_samples')
    if ws is not None and 'logl' in ws and 'points' in ws:
        ws_points = np.asarray(ws['points'], dtype=float)
        ws_logl = np.asarray(ws['logl'], dtype=float)
        if ws_points.shape == samples.shape and np.allclose(
            ws_points, samples, rtol=0, atol=0
        ):
            return ws_logl

    from .fitting import log_likelihood
    out = np.empty(len(samples), dtype=float)
    for i, theta in enumerate(samples):
        out[i] = log_likelihood(np.asarray(theta, dtype=float), spec)
    return out
