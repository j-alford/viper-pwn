"""Quick tests for the bin_style kwargs on the SED plotting functions."""

import os
import shutil
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pytest
import astropy.units as u

from viper.model import PWNModel
from viper.observables import Observable
from viper.fitting import build_fit_spec
from viper.plotting import (
    plot_sed,
    plot_model_sed,
    plot_sed_with_particles,
    plot_model_sed_movie,
    plot_electron_spectrum,
)


def _build_model():
    m = PWNModel()
    m.period.value        = 0.31686 * u.s
    m.p_dot.value         = 3.614e-13
    m.esn.value           = 1.0e51 * u.erg
    m.mej.value           = 6.0 * u.solMass
    m.n_ism.value         = 0.4 * u.Unit("1 / cm3")
    m.distance.value      = 1400.0 * u.pc
    m.braking_index.value = 2.5
    m.tau_sd.value        = 3000.0 * u.yr
    m.eta_b.value         = 0.01
    m.e_min.value         = 3.0 * u.GeV
    m.e_max.value         = 5e5 * u.GeV
    m.e_break.value       = 50.0 * u.GeV
    m.p1.value, m.p2.value = 0.8, 2.5
    # build_fit_spec needs one free parameter
    p1 = m._parameters['p1']
    p1.frozen = False
    p1._bounds = (0.0, 2.0)
    return m


@pytest.fixture(scope='module')
def plot_inputs():
    """Build a fit spec + theta + result usable by every plotting test."""
    m = _build_model()
    obs = [
        Observable.angular_size_snr(45.0, 10.0, 'arcmin'),
        Observable.angular_size_pwn(12.0, 6.0, 'arcmin'),
        Observable.flux_density(1.4, 'GHz', 1e-15, 1e-16, 'erg/s/cm2'),
    ]
    spec = build_fit_spec(m, obs)
    theta = list(m.parameters.free_fit_values)
    # movie needs snapshots
    result = m.run(dyn_step=20, elec_step=20, phot_step=20)
    return {'spec': spec, 'theta': theta, 'model': m, 'result': result}


@pytest.fixture(scope='module')
def escape_result():
    """Escape-on run with snapshots for the escaped-overlay tests."""
    m = _build_model()
    m.escape_model = 'bohm'
    m.parameters['esc_norm'].value = 1.0
    return m.run(dyn_step=20, elec_step=20, phot_step=20)


STYLES = ['curve', 'step', 'shaded_step', 'faded_step']


def _require_ffmpeg():
    """Locate a WORKING ffmpeg for movie tests, or skip."""
    ffmpeg = shutil.which('ffmpeg') or (
        '/opt/homebrew/bin/ffmpeg'
        if os.path.exists('/opt/homebrew/bin/ffmpeg') else None
    )
    if ffmpeg is None:
        pytest.skip('ffmpeg not available; cannot render movie')
    probe = subprocess.run([ffmpeg, '-version'], capture_output=True)
    if probe.returncode != 0:
        pytest.skip(f'ffmpeg at {ffmpeg} is present but not runnable '
                    f'(exit {probe.returncode}); cannot render movie')
    matplotlib.rcParams['animation.ffmpeg_path'] = ffmpeg
    return ffmpeg


@pytest.mark.parametrize('style', STYLES)
def test_plot_sed_bin_style(plot_inputs, style):
    fig, _ = plot_sed(
        plot_inputs['spec'], plot_inputs['theta'],
        show_residuals=False, phot_bin_style=style,
    )
    plt.close(fig)


@pytest.mark.parametrize('style', STYLES)
def test_plot_model_sed_bin_style(plot_inputs, style):
    fig, _ = plot_model_sed(
        plot_inputs['result'], 1400.0,
        phot_bin_style=style, elec_bin_style=style,
    )
    plt.close(fig)


@pytest.mark.parametrize('style', STYLES)
def test_plot_sed_with_particles_bin_style(plot_inputs, style):
    fig, _ = plot_sed_with_particles(
        plot_inputs['spec'], plot_inputs['theta'],
        show_residuals=False,
        phot_bin_style=style, elec_bin_style=style,
    )
    plt.close(fig)


@pytest.mark.parametrize('kwarg', ['phot_bin_style', 'elec_bin_style'])
def test_plot_model_sed_rejects_unknown(plot_inputs, kwarg):
    with pytest.raises(ValueError):
        plot_model_sed(
            plot_inputs['result'], 1400.0,
            **{kwarg: 'bogus'},
        )


def test_plot_sed_rejects_unknown(plot_inputs):
    with pytest.raises(ValueError):
        plot_sed(
            plot_inputs['spec'], plot_inputs['theta'],
            show_residuals=False, phot_bin_style='bogus',
        )


def test_plot_sed_with_particles_rejects_unknown(plot_inputs):
    with pytest.raises(ValueError):
        plot_sed_with_particles(
            plot_inputs['spec'], plot_inputs['theta'],
            show_residuals=False, elec_bin_style='bogus',
        )


@pytest.mark.parametrize('style', STYLES)
def test_plot_model_sed_movie_bin_style(plot_inputs, tmp_path, style):
    _require_ffmpeg()
    out = tmp_path / f'movie_{style}.mp4'
    plot_model_sed_movie(
        plot_inputs['result'], 1400.0, outfile=str(out),
        start_frame=0, end_frame=2,
        verbose=False,
        phot_bin_style=style, elec_bin_style=style, inj_bin_style=style,
    )
    assert out.exists() and out.stat().st_size > 0


def test_plot_model_sed_movie_debug_cooling(plot_inputs, tmp_path):
    _require_ffmpeg()
    out = tmp_path / 'movie_debug_cooling.mp4'
    plot_model_sed_movie(
        plot_inputs['result'], 1400.0, outfile=str(out),
        start_frame=0, end_frame=3,
        verbose=False, debug_cooling=True,
    )
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.parametrize('kwarg', ['phot_bin_style', 'elec_bin_style',
                                   'inj_bin_style', 'esc_bin_style'])
def test_plot_model_sed_movie_rejects_unknown(plot_inputs, tmp_path, kwarg):
    with pytest.raises(ValueError):
        plot_model_sed_movie(
            plot_inputs['result'], 1400.0,
            outfile=str(tmp_path / 'ignored.mp4'),
            start_frame=0, end_frame=1, verbose=False,
            **{kwarg: 'bogus'},
        )


# ---- escaped-particle ('halo') overlay ----

def test_plot_electron_spectrum_show_escaped(escape_result):
    """Standalone overlay adds a labeled 'Escaped (cumulative)' curve."""
    fig, ax = plot_electron_spectrum(escape_result, show_escaped=True)
    labels = [ln.get_label() for ln in ax.get_lines()]
    assert 'Escaped (cumulative)' in labels
    plt.close(fig)


def test_plot_electron_spectrum_show_escaped_off_raises(plot_inputs):
    with pytest.raises(ValueError):
        plot_electron_spectrum(plot_inputs['result'], show_escaped=True)


def test_plot_model_sed_movie_show_escaped(escape_result, tmp_path):
    """Movie renders with the escaped overlay (gif -> pillow, no ffmpeg)."""
    out = tmp_path / 'movie_escaped.gif'
    plot_model_sed_movie(
        escape_result, 1400.0, outfile=str(out),
        start_frame=0, end_frame=3, verbose=False, show_escaped=True,
    )
    assert out.exists() and out.stat().st_size > 0


def test_plot_model_sed_movie_show_escaped_off_raises(plot_inputs, tmp_path):
    with pytest.raises(ValueError):
        plot_model_sed_movie(
            plot_inputs['result'], 1400.0,
            outfile=str(tmp_path / 'ignored.gif'),
            start_frame=0, end_frame=2, verbose=False, show_escaped=True,
        )


def test_cooling_timescale_yr_constants():
    """_cooling_timescale_yr must match the (4/3) pitch-angle-averaged convention."""
    from viper.plotting import _cooling_timescale_yr

    # IC-only (B = 0): tau_IC(40 TeV, CMB) ~ 29,889 yr
    tau_ic = _cooling_timescale_yr(E_TeV=40.0, B_uG=0.0)
    assert 29000 < tau_ic < 31000, f"IC-only tau = {tau_ic:.0f}"

    # Sync-dominated (high B, 40 TeV): tau_sync = 1.253e7 / (B^2 * E)
    tau_sync_only = _cooling_timescale_yr(E_TeV=40.0, B_uG=100.0)
    # = 1.253e7 / (10000 * 40) = 31.3 yr
    assert 30 < tau_sync_only < 33, f"sync-only tau = {tau_sync_only:.1f}"

    # 1/tau = 1/29889 + 1/78313 -> tau ~ 21630 yr
    tau = _cooling_timescale_yr(E_TeV=40.0, B_uG=2.0)
    assert 21000 < tau < 22500, f"combined tau = {tau:.0f}"

    # tau halves when E doubles
    tau_80 = _cooling_timescale_yr(E_TeV=80.0, B_uG=2.0)
    assert abs(tau_80 - tau / 2.0) / tau_80 < 1e-6, (
        f"non-linear scaling: tau(40)={tau:.0f}, tau(80)={tau_80:.0f}")
