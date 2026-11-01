"""fitting.starting_positions: scatter ball and explicit walker files."""
import numpy as np
import pytest

from viper import fitting as F
from test_likelihood import _make_spec


def _spec():
    spec, _ = _make_spec([])
    return spec


def test_scatter_ball_inside_bounds():
    spec = _spec()
    rng = np.random.default_rng(1)
    p0 = F.starting_positions(spec, [0.0], 16, 0.01, rng)
    assert p0.shape == (16, 1)
    lo, hi = spec.free_params[0].bounds
    assert np.all(p0 > lo) and np.all(p0 < hi)
    assert all(np.isfinite(F.log_prior(w, spec)) for w in p0)


def test_walkers_returned_clipped():
    spec = _spec()
    lo, hi = spec.free_params[0].bounds
    walkers = np.array([[-0.5], [0.3], [hi + 1.0]])
    p0 = F.starting_positions(spec, None, 3, 0.1, np.random.default_rng(0),
                              walkers=walkers)
    assert p0[0, 0] == -0.5 and p0[1, 0] == 0.3
    assert p0[2, 0] < hi


def test_walkers_shape_mismatch():
    spec = _spec()
    with pytest.raises(ValueError, match="shape"):
        F.starting_positions(spec, None, 4, 0.1, np.random.default_rng(0),
                             walkers=np.zeros((3, 1)))


def test_walkers_outside_prior_refused(monkeypatch):
    spec = _spec()
    monkeypatch.setattr(F, 'log_prior',
                        lambda th, s: -np.inf if th[0] > 0.5 else 0.0)
    walkers = np.array([[-0.5], [0.9], [0.0], [0.7]])
    with pytest.raises(ValueError, match=r"rows outside the prior: \[1, 3\]"):
        F.starting_positions(spec, None, 4, 0.1, np.random.default_rng(0),
                             walkers=walkers)


def test_scatter_redraws_then_gives_up(monkeypatch):
    spec = _spec()
    monkeypatch.setattr(F, 'log_prior', lambda th, s: -np.inf)
    with pytest.raises(ValueError, match="after 3 redraws"):
        F.starting_positions(spec, [0.0], 8, 0.01, np.random.default_rng(0),
                             max_redraws=3)
