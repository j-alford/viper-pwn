"""The neutron-star moment of inertia parameter ``i_ns`` (10^45 g cm^2)."""

import numpy as np
import pytest

from viper.pulsar import e0_37_from_timing
from viper.model import PWNModel
from viper.scalar_params import PWNScalarParams, check_physical_inputs

P, PDOT, N, TAU = 0.0334, 4.2e-13, 2.5, 700.0


def test_e0_scales_linearly():
    base = e0_37_from_timing(P, PDOT, N, TAU)
    assert e0_37_from_timing(P, PDOT, N, TAU, 1.0) == base
    assert np.isclose(e0_37_from_timing(P, PDOT, N, TAU, 2.5), 2.5 * base)


def test_model_default_is_one_with_bounds():
    import astropy.units as u
    m = PWNModel()
    par = m.parameters['i_ns']
    assert par.value == 1.0                     # internal: 10^45 g cm^2
    assert par.frozen and par.log
    assert par.bounds == (0.3, 3.0)
    m.i_ns.value = 2e45 * u.g * u.cm**2         # physical units in, internal out
    assert np.isclose(m.i_ns.value, 2.0)
    m.i_ns.bounds = (3e44 * u.g * u.cm**2, 3e45 * u.g * u.cm**2)
    assert np.allclose(m.i_ns.bounds, (0.3, 3.0))


def test_model_e0_and_esd_follow_the_multiplier():
    m = PWNModel()
    m.period._set_internal(P)
    m.p_dot._set_internal(PDOT)
    m.braking_index.value = N
    m.tau_sd._set_internal(TAU)
    e0, esd = m.e0_37, m.esd_40
    m.i_ns._set_internal(2.0)
    assert np.isclose(m.e0_37, 2 * e0)
    assert np.isclose(m.esd_40, 2 * esd)


def test_scalar_params_default_and_scaling():
    kw = dict(esn=1.0, mej=8.0, n_ism=0.1, braking_index=N, tau_sd=TAU,
              period=P, p_dot=PDOT, eta_b=0.03, e_min=0.5, e_max=4e6,
              e_break=1000.0, p1=1.6, p2=2.5)
    a = PWNScalarParams(**kw)
    b = PWNScalarParams(i_ns=3.0, **kw)
    assert a.i_ns == 1.0
    assert np.isclose(b.e0_37, 3 * a.e0_37)


def test_nonpositive_rejected():
    kw = dict(esn=1.0, mej=8.0, n_ism=0.1, braking_index=N, tau_sd=TAU,
              period=P, p_dot=PDOT, eta_b=0.03, eta_g=0.0, e_min=0.5,
              e_max=4e6, e_break=1000.0, p1=1.6, p2=2.5, f_max=0.0,
              vel_psr=0.0)
    assert check_physical_inputs(i_ns=1.0, **kw) == []
    assert any('i_ns' in s for s in check_physical_inputs(i_ns=0.0, **kw))


def test_fit_spec_carries_it(tmp_path):
    from viper.fitting import build_fit_spec, _theta_to_evolve_args
    m = PWNModel()
    m.period._set_internal(P)
    m.p_dot._set_internal(PDOT)
    m.braking_index.value = N
    m.tau_sd._set_internal(TAU)
    import astropy.units as u
    m.esn.frozen = False
    m.esn.bounds = (0.1e51 * u.erg, 5e51 * u.erg)
    spec = build_fit_spec(m, [])
    args1 = _theta_to_evolve_args(np.array([np.log10(1.0)]), spec)
    m.i_ns._set_internal(2.0)
    spec2 = build_fit_spec(m, [])
    args2 = _theta_to_evolve_args(np.array([np.log10(1.0)]), spec2)
    assert np.isclose(args2['e0_37'], 2 * args1['e0_37'])
