"""Source files: one Python file that describes a fit.

Pins (a) what each block of a source file does to the model and fit
spec, (b) that mistakes are refused with a message naming the block,
(c) the run-folder layout, and (d) the runner end to end.
"""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

import viper.constants as C
from viper.source import (
    SNAPSHOT_NAME, SourceError, build_from_source, load_run, load_source,
    new_run_dir, source_settings, start_run,
)

ROOT = Path(__file__).resolve().parents[1]

BASE = """
    import astropy.units as u
    from viper import Observable

    PULSAR = 'J1930+1852'

    OBSERVABLES = [
        Observable.angular_size_pwn(1.14, 0.04, 'arcmin'),
        Observable.flux_density(1.4, 'GHz', 433, 30, 'mJy', weight=3.0),
        Observable.band_flux(2, 10, 'keV', 5.43e-12, 0.035e-12),
    ]

    PARAMETERS = {
        'mej':   (21 * u.M_sun, 1 * u.M_sun, 100 * u.M_sun),
        'eta_b': (0.0024, 1e-5, 0.5),
    }

    FROZEN = {
        'esn':           0.95e51 * u.erg,
        'n_ism':         0.008 / u.cm**3,
        'braking_index': 1.9,
        'tau_sd':        3574 * u.yr,
        'e_min':         10 * u.GeV,
        'e_max':         400 * u.TeV,
        'e_break':       3232 * u.GeV,
        'p1':            2.29,
        'p2':            2.42,
        'distance':      6.2 * u.kpc,
    }
"""


def _write(tmp_path, extra='', base=BASE, name='mysrc.py'):
    path = tmp_path / name
    path.write_text(textwrap.dedent(base) + textwrap.dedent(extra))
    return str(path)


def _build(tmp_path, extra='', base=BASE):
    return build_from_source(load_source(_write(tmp_path, extra, base)))


# What the blocks do

def test_example_file_builds():
    model, obs, spec = build_from_source(
        load_source(str(ROOT / 'examples' / 'g54.py')))
    assert spec.param_names == [
        'mej', 'n_ism', 'tau_sd', 'eta_b', 'e_break', 'distance']
    assert len(obs) == 7
    assert spec.ndim < len(obs)
    assert model.parameters['distance'].prior == (6200.0, 1000.0)
    assert spec.n_ic == 1
    assert model.parameters['ic_temp_1'].frozen
    assert model.parameters['e_max'].value == pytest.approx(4e5)


def test_parameters_and_frozen(tmp_path):
    model, obs, spec = _build(tmp_path)
    assert spec.param_names == ['mej', 'eta_b']
    mej = model.parameters['mej']
    assert (mej.value, mej.bounds, mej.frozen) == (21.0, (1.0, 100.0), False)
    assert model.parameters['esn'].value == pytest.approx(0.95)
    assert model.parameters['esn'].frozen
    assert model.parameters['distance'].value == pytest.approx(6200.0)
    assert [ob.weight for ob in obs] == [1.0, 3.0, 1.0]


def test_defaults(tmp_path):
    src = load_source(_write(tmp_path))
    _, _, spec = build_from_source(src)
    settings = source_settings(src)
    assert settings['FIT']['nwalkers'] == 96
    assert settings['FIT']['nsteps'] == 1000
    assert settings['NESTED']['nlive'] == 800
    assert spec.integration == 'pc_full'
    assert spec.use_cmb_lookup is True
    assert spec.snapshot_mode == 'secant'


def test_options_and_fit_blocks(tmp_path):
    extra = """
        OPTIONS = {'integration': 'euler', 'snapshot_mode': 'instantaneous',
                   'use_cmb_lookup': False}
        FIT = {'nwalkers': 8, 'nsteps': 5, 'seed': 7}
    """
    src = load_source(_write(tmp_path, extra))
    _, _, spec = build_from_source(src)
    assert spec.integration == 'euler'
    assert spec.snapshot_mode == 'instantaneous'
    assert spec.use_cmb_lookup is False
    fit = source_settings(src)['FIT']
    assert (fit['nwalkers'], fit['nsteps'], fit['seed']) == (8, 5, 7)


def test_period_and_pdot_instead_of_pulsar(tmp_path):
    base = BASE.replace("PULSAR = 'J1930+1852'",
                        "PERIOD = 0.137 * u.s\n    P_DOT = 7.5e-13")
    model, _, spec = _build(tmp_path, base=base)
    assert spec.period == pytest.approx(0.137)
    assert spec.p_dot == pytest.approx(7.5e-13)


def test_priors(tmp_path):
    extra = """
        PRIORS = {'mej': (12 * u.M_sun, 4 * u.M_sun),
                  'eta_b': (-2.0, 1.0, 'log')}
    """
    model, _, spec = _build(tmp_path, extra)
    by_name = {p.name: p for p in spec.free_params}
    assert by_name['mej'].prior == (12.0, 4.0)
    assert by_name['mej'].prior_log is False
    assert by_name['eta_b'].prior == (-2.0, 1.0)
    assert by_name['eta_b'].prior_log is True


def test_prior_needs_units(tmp_path):
    with pytest.raises(ValueError, match='astropy Quantity'):
        _build(tmp_path, "PRIORS = {'mej': (12, 4)}")


def test_linear_override(tmp_path):
    base = BASE.replace("'eta_b': (0.0024, 1e-5, 0.5),",
                        "'eta_b': (0.0024, 1e-5, 0.5, 'linear'),")
    _, _, spec = _build(tmp_path, base=base)
    by_name = {p.name: p for p in spec.free_params}
    assert by_name['eta_b'].log is False
    assert by_name['eta_b'].bounds == (1e-5, 0.5)
    assert by_name['mej'].log is True


def test_ic_fields(tmp_path):
    extra = """
        IC_FIELDS = [
            {'temp': 35 * u.K, 'norm': 0.35 * u.eV / u.cm**3},
            {'temp': 3000, 'norm': 0.5, 'norm_bounds': (0.1, 5.0)},
        ]
    """
    model, _, spec = _build(tmp_path, extra)
    assert spec.n_ic == 2
    assert spec.ic_temps == [35.0, 3000.0]
    assert spec.ic_norms == [0.35, 0.5]
    assert spec.param_names == ['mej', 'eta_b', 'ic_norm_2']
    assert model.parameters['ic_norm_2'].bounds == (0.1, 5.0)
    assert model.parameters['ic_temp_2'].frozen
    assert model.parameters['ic_norm_1'].frozen


# Mistakes are refused

@pytest.mark.parametrize('extra, match', [
    ("OPTIONS = {'integraton': 'euler'}", "unknown OPTIONS key"),
    ("FIT = {'nwalker': 8}", "unknown FIT key"),
    ("NESTED = {'nlives': 8}", "unknown NESTED key"),
    ("PRIOR = {'mej': (12 * u.M_sun, 4 * u.M_sun)}", "misspelling of 'PRIORS'"),
    ("PRIORS = {'mass': (12, 4)}", "unknown parameter 'mass'"),
    ("FROZEN['mej'] = 5 * u.M_sun", "both PARAMETERS and FROZEN"),
    ("PARAMETERS['p1'] = (2.0, 0.5)", "both PARAMETERS and FROZEN"),
    ("del FROZEN['p2']", r"\['p2'\] must appear"),
    ("PARAMETERS['mej'] = (200 * u.M_sun, 1 * u.M_sun, 100 * u.M_sun)",
     "outside its bounds"),
    ("PARAMETERS['mej'] = (21 * u.M_sun, 1 * u.M_sun)", "must be"),
    ("PARAMETERS['mej'] = (21, 1, 100)", "astropy Quantity"),
    ("OBSERVABLES = []", "OBSERVABLES is missing or empty"),
    ("OBSERVABLES = [1.0]", "list of viper.Observable"),
    ("IC_FIELDS = [{'temp': 35}]", "needs 'temp' and 'norm'"),
    ("IC_FIELDS = [{'temp': 35, 'norm': 0.3, 'frozen': True}]",
     "needs 'temp' and 'norm'"),
])
def test_mistakes(tmp_path, extra, match):
    with pytest.raises(ValueError, match=match):
        _build(tmp_path, extra)


def test_own_tables_are_allowed(tmp_path):
    base = BASE.replace("PULSAR =", "CXO_DATA = [(1.0, 2.0)]\n    PULSAR =")
    _build(tmp_path, base=base)


def test_pulsar_required(tmp_path):
    base = BASE.replace("PULSAR = 'J1930+1852'", "")
    with pytest.raises(SourceError, match='PULSAR'):
        _build(tmp_path, base=base)


def test_distance_required_when_catalogue_has_none(tmp_path):
    base = BASE.replace("PULSAR = 'J1930+1852'", "PULSAR = 'J0359+5414'")
    with pytest.raises(SourceError, match="no distance for J0359\\+5414"):
        _build(tmp_path, "del FROZEN['distance']", base=base)


def test_distance_required_without_catalogue(tmp_path):
    base = BASE.replace("PULSAR = 'J1930+1852'",
                        "PERIOD = 0.137 * u.s\n    P_DOT = 7.5e-13")
    with pytest.raises(SourceError, match=r"\['distance'\] must appear"):
        _build(tmp_path, "del FROZEN['distance']", base=base)


def test_constants_mismatch(tmp_path):
    other = 'astropy' if C.CONSTANTS_SOURCE != 'astropy' else 'gsz2009'
    with pytest.raises(SourceError, match=f'VIPER_CONSTANTS={other}'):
        _build(tmp_path, f"OPTIONS = {{'constants': '{other}'}}")
    _build(tmp_path, f"OPTIONS = {{'constants': '{C.CONSTANTS_SOURCE}'}}")


def test_missing_file(tmp_path):
    with pytest.raises(SourceError, match='not found'):
        load_source(str(tmp_path / 'nope.py'))


# Run folders

def test_run_folder(tmp_path):
    path = _write(tmp_path)
    root = str(tmp_path / 'results')
    first = new_run_dir(path, root=root)
    assert first == os.path.join(root, 'mysrc_run01')
    copy = start_run(path, first)
    assert os.path.basename(copy) == SNAPSHOT_NAME
    assert open(copy).read() == open(path).read()
    assert new_run_dir(path, root=root) == os.path.join(root, 'mysrc_run02')
    os.makedirs(os.path.join(root, 'mysrc_run07_test'))
    os.makedirs(os.path.join(root, 'other_run12'))
    assert new_run_dir(path, root=root) == os.path.join(root, 'mysrc_run08')

    # the copy, not the original, defines the run
    Path(path).write_text(open(path).read().replace('21 * u.M_sun',
                                                     '30 * u.M_sun'))
    model, _, _ = build_from_source(load_run(first))
    assert model.parameters['mej'].value == 21.0
    assert not os.path.exists(os.path.join(first, '__pycache__'))


def test_load_run_needs_a_snapshot(tmp_path):
    with pytest.raises(SourceError, match=SNAPSHOT_NAME):
        load_run(str(tmp_path))


def test_workers_load_the_source_file(tmp_path, monkeypatch):
    from viper import fitting
    from viper.injection import INJECTION_MODELS
    extra = """
        from viper.injection import register_injection_model

        def _flat(energy, e_break, p1, p2):
            return energy ** -2.0

        register_injection_model('test_source_flat', _flat)
    """
    path = _write(tmp_path, extra)
    INJECTION_MODELS.pop('test_source_flat', None)
    monkeypatch.setenv('VIPER_SOURCE_FILE', path)
    try:
        fitting._init_worker(None)
        assert 'test_source_flat' in INJECTION_MODELS
    finally:
        INJECTION_MODELS.pop('test_source_flat', None)


# The runner, through its command line

def _run_tool(tool, *args, cwd):
    env = dict(os.environ, NUMBA_NUM_THREADS='1')
    return subprocess.run(
        [sys.executable, str(ROOT / 'tools' / tool), *args],
        cwd=cwd, env=env, capture_output=True, text=True, timeout=900)


@pytest.mark.slow
def test_runner_end_to_end(tmp_path):
    import emcee
    path = _write(tmp_path, "FIT = {'nwalkers': 4, 'nsteps': 1}")

    out = _run_tool('run_mcmc.py', path, '--ncores', '1', cwd=tmp_path)
    assert out.returncode == 0, out.stdout + out.stderr
    run = tmp_path / 'results' / 'mysrc_run01'
    assert sorted(p.name for p in run.iterdir()) == [
        'best_theta.npy', 'chain.h5', SNAPSHOT_NAME]
    assert np.load(run / 'best_theta.npy').shape == (2,)

    out = _run_tool('run_mcmc.py', str(run), '--extend', '1',
                    '--ncores', '1', cwd=tmp_path)
    assert out.returncode == 0, out.stdout + out.stderr
    backend = emcee.backends.HDFBackend(str(run / 'chain.h5'),
                                        read_only=True)
    assert backend.iteration == 2

    out = _run_tool('monitor.py', str(run / 'chain.h5'), '--plain',
                    cwd=tmp_path)
    assert out.returncode == 0, out.stdout + out.stderr
    assert 'Steps: 2' in out.stdout
    assert 'F(1.4 GHz) [mJy]' in out.stdout


def test_runner_refuses_a_bad_source_file(tmp_path):
    path = _write(tmp_path, "FIT = {'nwalker': 4}")
    out = _run_tool('run_mcmc.py', path, cwd=tmp_path)
    assert out.returncode == 2
    assert 'unknown FIT key' in out.stdout
    assert not (tmp_path / 'results').exists()


def test_runner_refuses_an_old_style_folder(tmp_path):
    (tmp_path / 'run_config.json').write_text('{}')
    out = _run_tool('run_mcmc.py', str(tmp_path), cwd=tmp_path)
    assert out.returncode == 2
    assert SNAPSHOT_NAME in out.stdout
