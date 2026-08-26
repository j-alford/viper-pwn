"""Nested-sampling tests: prior_transform, runner selection, chain adapter."""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from viper.fitting import _ParamSpec, _FitSpec
from viper.nested import (
    prior_transform,
    prior_transform_vec,
    run_nested,
    write_emcee_chain,
)


# ===================================================================
# Helpers
# ===================================================================

def _make_spec(param_specs):
    """Wrap a list of _ParamSpec in a _FitSpec stub."""
    return _FitSpec(
        free_params=list(param_specs),
        all_values={},
        period=0.0,
        p_dot=0.0,
        n_ic=0,
        ic_temps=[],
        ic_norms=[],
        ic_temp_names=[],
        ic_norm_names=[],
    )


# module-level for Pool pickling
_TOY_TRUTH = np.array([0.5, -0.3])
_TOY_SIGMA = 0.1


def _toy_loglike(theta):
    """2-D isotropic Gaussian loglike (picklable, no closures)."""
    theta = np.asarray(theta, dtype=float)
    d = (theta - _TOY_TRUTH) / _TOY_SIGMA
    return float(-0.5 * np.dot(d, d))


# ===================================================================
# 1-5: prior_transform unit tests
# ===================================================================

def test_prior_transform_uniform_linear():
    spec = _make_spec([
        _ParamSpec(name='x', log=False, bounds=(0., 10.), prior=None,
                   prior_log=False),
    ])
    assert prior_transform(np.array([0.0]), spec)[0] == pytest.approx(0.0)
    assert prior_transform(np.array([0.5]), spec)[0] == pytest.approx(5.0)
    assert prior_transform(np.array([1.0]), spec)[0] == pytest.approx(10.0)


def test_prior_transform_uniform_log():
    lo, hi = np.log10(1e-4), np.log10(0.5)
    spec = _make_spec([
        _ParamSpec(name='y', log=True, bounds=(lo, hi), prior=None,
                   prior_log=False),
    ])
    out = prior_transform(np.array([0.5]), spec)[0]
    expected = 0.5 * (lo + hi)
    assert out == pytest.approx(expected, abs=0.01)


def test_prior_transform_gaussian_linear():
    spec = _make_spec([
        _ParamSpec(name='d', log=False, bounds=(400., 3000.),
                   prior=(1400., 400.), prior_log=False),
    ])
    out = prior_transform(np.array([0.5]), spec)[0]
    assert abs(out - 1400.) < 5.0


def test_prior_transform_gaussian_log_param():
    lo, hi = np.log10(0.5), np.log10(100.)
    spec = _make_spec([
        _ParamSpec(name='m', log=True, bounds=(lo, hi),
                   prior=(10., 5.), prior_log=False),
    ])
    out = prior_transform(np.array([0.5]), spec)[0]
    assert abs(out - np.log10(10.)) < 0.05


def test_prior_transform_bounds_respected():
    """All sampled thetas must lie strictly within the declared fit bounds."""
    fps = [
        _ParamSpec(name='x', log=False, bounds=(0., 10.), prior=None,
                   prior_log=False),
        _ParamSpec(name='y', log=True, bounds=(np.log10(1e-4), np.log10(0.5)),
                   prior=None, prior_log=False),
        _ParamSpec(name='d', log=False, bounds=(400., 3000.),
                   prior=(1400., 400.), prior_log=False),
        _ParamSpec(name='m', log=True, bounds=(np.log10(0.5), np.log10(100.)),
                   prior=(10., 5.), prior_log=False),
        _ParamSpec(name='lp', log=True, bounds=(0., 6.),
                   prior=(3., 1.), prior_log=True),
    ]
    spec = _make_spec(fps)
    rng = np.random.default_rng(0)
    U = rng.uniform(0, 1, size=(1000, len(fps)))
    Theta = prior_transform_vec(U, spec)
    for i, fp in enumerate(fps):
        lo, hi = fp.bounds
        assert np.all(Theta[:, i] >= lo - 1e-9), \
            f"param {fp.name} has values below lower bound {lo}"
        assert np.all(Theta[:, i] <= hi + 1e-9), \
            f"param {fp.name} has values above upper bound {hi}"


# ===================================================================
# 6: Integration: toy 2D Gaussian recovery
# ===================================================================

def _toy_2d_spec():
    """Mock _FitSpec for the 2-D Gaussian recovery tests."""
    return _make_spec([
        _ParamSpec(name='a', log=False, bounds=(-3., 3.), prior=None,
                   prior_log=False),
        _ParamSpec(name='b', log=False, bounds=(-3., 3.), prior=None,
                   prior_log=False),
    ])


@pytest.mark.slow
def test_2d_gaussian_recovery():
    """UltraNest recovers the mean of an isotropic 2D Gaussian likelihood."""
    pytest.importorskip("ultranest")
    import ultranest
    import ultranest.stepsampler

    spec = _toy_2d_spec()

    def transform(u):
        return prior_transform(np.asarray(u, dtype=float), spec)

    with tempfile.TemporaryDirectory() as tmp:
        sampler = ultranest.ReactiveNestedSampler(
            param_names=['a', 'b'],
            loglike=_toy_loglike,
            transform=transform,
            log_dir=tmp,
            resume='overwrite',
        )
        sampler.stepsampler = ultranest.stepsampler.SliceSampler(
            nsteps=8,
            generate_direction=ultranest.stepsampler.generate_mixture_random_direction,
        )
        result = sampler.run(min_num_live_points=200, dlogz=0.5,
                             show_status=False, viz_callback=None)

    samples = np.asarray(result['samples'])
    posterior_mean = samples.mean(axis=0)
    posterior_std = samples.std(axis=0)
    for k in range(2):
        assert abs(posterior_mean[k] - _TOY_TRUTH[k]) < 3 * posterior_std[k] + 0.05, (
            f"posterior mean[{k}]={posterior_mean[k]:.3f} not within "
            f"3 sigma of truth={_TOY_TRUTH[k]} (sigma={posterior_std[k]:.3f})"
        )
    assert np.isfinite(result['logz'])
    assert np.isfinite(result['logzerr'])


# ===================================================================
# vectorized parallel via multiprocessing.Pool
# ===================================================================

def test_log_like_worker_dispatches_to_log_likelihood(monkeypatch):
    """_log_like_worker must read _worker_spec and forward to log_likelihood."""
    from viper import fitting

    fake_spec = object()
    calls = []

    def fake_loglike(theta, spec):
        calls.append((np.asarray(theta), spec))
        return -42.5

    monkeypatch.setattr(fitting, 'log_likelihood', fake_loglike)
    monkeypatch.setattr(fitting, '_worker_spec', fake_spec)

    theta = np.array([1.0, 2.0, 3.0])
    out = fitting._log_like_worker(theta)

    assert out == -42.5
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], theta)
    assert calls[0][1] is fake_spec


def test_log_like_worker_is_picklable():
    """Module-level reference must round-trip through pickle for Pool workers."""
    import pickle
    from viper.fitting import _log_like_worker

    pickled = pickle.dumps(_log_like_worker)
    unpickled = pickle.loads(pickled)
    assert unpickled is _log_like_worker


def test_run_nested_signature_includes_n_cores():
    """n_cores must be a recognized keyword on run_nested with default 1."""
    import inspect
    sig = inspect.signature(run_nested)
    assert 'n_cores' in sig.parameters
    assert sig.parameters['n_cores'].default == 1


def test_run_ultranest_dispatches_on_n_cores(monkeypatch):
    """n_cores=1 -> _run_ultranest_serial; n_cores>1 -> _run_ultranest_parallel."""
    from viper import nested as nm
    from viper import fitting as fm

    fake_spec = _toy_2d_spec()

    class FakeModel:
        def validate(self):
            pass

    monkeypatch.setattr(fm, '_build_fit_spec', lambda *a, **kw: fake_spec)

    captured = []

    def fake_serial(spec, **kwargs):
        captured.append(('serial', kwargs.get('nlive')))
        return ({'samples': np.zeros((1, len(spec.free_params)))}, spec)

    def fake_parallel(spec, **kwargs):
        captured.append(('parallel', kwargs.get('n_cores')))
        return ({'samples': np.zeros((1, len(spec.free_params)))}, spec)

    monkeypatch.setattr(nm, '_run_ultranest_serial', fake_serial)
    monkeypatch.setattr(nm, '_run_ultranest_parallel', fake_parallel)

    nm._run_ultranest(
        FakeModel(), [], integration='pc_full',
        nlive=42, dlogz=0.5, slice_steps=4, n_cores=1,
        log_dir=None, resume='overwrite', show_status=False,
    )
    assert captured == [('serial', 42)]

    captured.clear()
    nm._run_ultranest(
        FakeModel(), [], integration='pc_full',
        nlive=42, dlogz=0.5, slice_steps=4, n_cores=4,
        log_dir=None, resume='overwrite', show_status=False,
    )
    assert captured == [('parallel', 4)]


@pytest.mark.slow
def test_vectorized_2d_gaussian_recovery():
    """Pool + UltraNest vectorized=True path, as _run_ultranest_parallel uses."""
    pytest.importorskip("ultranest")
    import multiprocessing
    import ultranest
    import ultranest.stepsampler

    spec = _toy_2d_spec()

    def vec_transform(u_batch):
        u_batch = np.asarray(u_batch, dtype=float)
        return np.array([prior_transform(u, spec) for u in u_batch])

    pool = multiprocessing.Pool(2)
    try:
        def vec_loglike(theta_batch):
            theta_batch = np.asarray(theta_batch, dtype=float)
            results = pool.map(_toy_loglike, list(theta_batch))
            return np.asarray(results, dtype=float)

        with tempfile.TemporaryDirectory() as tmp:
            sampler = ultranest.ReactiveNestedSampler(
                param_names=['a', 'b'],
                loglike=vec_loglike,
                transform=vec_transform,
                log_dir=tmp,
                resume='overwrite',
                vectorized=True,
            )
            sampler.stepsampler = ultranest.stepsampler.SliceSampler(
                nsteps=8,
                generate_direction=ultranest.stepsampler.generate_mixture_random_direction,
            )
            result = sampler.run(min_num_live_points=200, dlogz=0.5,
                                 show_status=False, viz_callback=None)
    finally:
        pool.close()
        pool.join()

    samples = np.asarray(result['samples'])
    posterior_mean = samples.mean(axis=0)
    posterior_std = samples.std(axis=0)
    for k in range(2):
        assert abs(posterior_mean[k] - _TOY_TRUTH[k]) < 3 * posterior_std[k] + 0.05, (
            f"posterior mean[{k}]={posterior_mean[k]:.3f} not within "
            f"3 sigma of truth={_TOY_TRUTH[k]} (sigma={posterior_std[k]:.3f})"
        )
    assert np.isfinite(result['logz'])
    assert np.isfinite(result['logzerr'])


# ===================================================================
# 7: chain.h5 round-trip
# ===================================================================

def test_chain_h5_round_trip():
    """write_emcee_chain output must be readable by emcee.HDFBackend."""
    pytest.importorskip("emcee")
    import emcee

    rng = np.random.default_rng(42)
    ndim = 4
    n_total = 4000
    samples = rng.standard_normal((n_total, ndim))
    loglikes = -0.5 * (samples ** 2).sum(axis=1)

    with tempfile.TemporaryDirectory() as tmp:
        h5path = os.path.join(tmp, 'chain.h5')
        write_emcee_chain(samples, loglikes, ndim, h5path, nwalkers=20)

        backend = emcee.backends.HDFBackend(h5path, read_only=True)
        chain = backend.get_chain()
        log_prob = backend.get_log_prob()

        nsteps_expected = n_total // 20
        assert chain.shape == (nsteps_expected, 20, ndim)
        assert log_prob.shape == (nsteps_expected, 20)

        flat_chain = chain.reshape(-1, ndim)
        flat_lp = log_prob.reshape(-1)
        np.testing.assert_allclose(flat_chain, samples[:nsteps_expected * 20])
        np.testing.assert_allclose(flat_lp, loglikes[:nsteps_expected * 20])


def test_chain_h5_too_few_samples_raises():
    rng = np.random.default_rng(0)
    samples = rng.standard_normal((10, 3))
    loglikes = np.zeros(10)
    with tempfile.TemporaryDirectory() as tmp:
        h5path = os.path.join(tmp, 'chain.h5')
        with pytest.raises(ValueError, match="Too few samples"):
            write_emcee_chain(samples, loglikes, ndim=3, h5path=h5path,
                              nwalkers=32)
