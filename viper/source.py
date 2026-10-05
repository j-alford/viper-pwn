"""
Source files: one Python file that describes a fit.

A source file is plain Python.  It defines these module-level names:

===============  ====================================================
``PULSAR``       ATNF pulsar name (or ``PERIOD`` and ``P_DOT``)
``OBSERVABLES``  list of :class:`~viper.observables.Observable`
``PARAMETERS``   free parameters, ``name: (start, low, high)``
``FROZEN``       fixed parameters, ``name: value`` (optional)
``PRIORS``       Gaussian priors, ``name: (center, sigma)`` (optional)
``IC_FIELDS``    extra inverse-Compton photon fields (optional)
``OPTIONS``      model switches (optional)
``FIT``          MCMC settings (optional)
``NESTED``       nested-sampling settings (optional)
===============  ====================================================

``examples/g54.py`` is a complete, commented example.  Values with a
physical dimension are astropy Quantities.

Each fit goes in its own run folder.
"""

from __future__ import annotations

import difflib
import importlib.util
import os
import re
import shutil
import sys

from . import constants as C
from .fitting import build_fit_spec
from .model import PWNModel
from .observables import Observable

SNAPSHOT_NAME = 'source.py'

SOURCE_NAMES = (
    'PULSAR', 'PERIOD', 'P_DOT', 'OBSERVABLES', 'PARAMETERS', 'FROZEN',
    'PRIORS', 'IC_FIELDS', 'OPTIONS', 'FIT', 'NESTED',
)

# Every model parameter must be given, in PARAMETERS or FROZEN.
REQUIRED_PARAMETERS = (
    'esn', 'mej', 'n_ism', 'braking_index', 'tau_sd', 'eta_b',
    'e_min', 'e_max', 'e_break', 'p1', 'p2',
)

OPTION_DEFAULTS = {
    'integration': 'pc_full',
    'constants': None,
    'use_cmb_lookup': True,
    'timestep_events': False,
    'snapshot_mode': 'secant',
    'event_dt_floor_yr': 0.5,
    'timestep_cooling': 'compat',
    'escape_model': 'off',
    'injection_model': 'broken_powerlaw',
    'ssc_model': 'off',
}

FIT_DEFAULTS = {
    'nwalkers': 96,
    'nsteps': 1000,
    'scatter': 5e-3,
    'seed': 42,
    'init_from': None,
}

NESTED_DEFAULTS = {
    'nlive': 800,
    'dlogz': 0.5,
    'slice_steps': None,
    'resume': 'resume',
}

_IC_KEYS = ('temp', 'norm', 'temp_bounds', 'norm_bounds')


class SourceError(ValueError):
    """A source file is missing something or contains a mistake."""


# Loading

def load_source(path: str):
    """Import a source file by path and return it as a module."""
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise SourceError(f"source file not found: {path}")
    stem = os.path.splitext(os.path.basename(path))[0]
    name = 'viper_source_' + re.sub(r'\W', '_', stem)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # keep run folders free of __pycache__
    dont_write = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = dont_write
    return module


def _blocks(src) -> dict:
    """Collect and check the blocks a source file defines."""
    given = {k: v for k, v in vars(src).items()
             if k.isupper() and not k.startswith('_')}
    label = getattr(src, '__file__', None) or 'source'

    # A misspelt block would otherwise be ignored without a word.
    for name in given:
        if name in SOURCE_NAMES:
            continue
        close = difflib.get_close_matches(name, SOURCE_NAMES, n=1, cutoff=0.8)
        if close:
            raise SourceError(
                f"{label}: '{name}' looks like a misspelling of "
                f"'{close[0]}'. Rename it (or give your own table a "
                f"more distinct name).")

    out = {k: given.get(k) for k in SOURCE_NAMES}
    if not out['OBSERVABLES']:
        raise SourceError(f"{label}: OBSERVABLES is missing or empty.")
    if not isinstance(out['OBSERVABLES'], (list, tuple)) or not all(
            isinstance(ob, Observable) for ob in out['OBSERVABLES']):
        raise SourceError(
            f"{label}: OBSERVABLES must be a list of viper.Observable.")
    if not out['PARAMETERS']:
        raise SourceError(
            f"{label}: PARAMETERS is missing or empty; a fit needs at "
            f"least one free parameter.")
    if out['PULSAR'] is None and (
            out['PERIOD'] is None or out['P_DOT'] is None):
        raise SourceError(
            f"{label}: give PULSAR (an ATNF name) or both PERIOD and "
            f"P_DOT.")

    for key, defaults in (('OPTIONS', OPTION_DEFAULTS),
                          ('FIT', FIT_DEFAULTS),
                          ('NESTED', NESTED_DEFAULTS)):
        block = dict(out[key] or {})
        unknown = sorted(set(block) - set(defaults))
        if unknown:
            raise SourceError(
                f"{label}: unknown {key} key(s) {unknown}. "
                f"Known keys: {sorted(defaults)}.")
        out[key] = {**defaults, **block}

    out['FROZEN'] = dict(out['FROZEN'] or {})
    out['PRIORS'] = dict(out['PRIORS'] or {})
    out['IC_FIELDS'] = list(out['IC_FIELDS'] or [])
    out['PARAMETERS'] = dict(out['PARAMETERS'])

    both = sorted(set(out['PARAMETERS']) & set(out['FROZEN']))
    if both:
        raise SourceError(
            f"{label}: {both} appear in both PARAMETERS and FROZEN.")
    named = set(out['PARAMETERS']) | set(out['FROZEN'])
    missing = [p for p in REQUIRED_PARAMETERS if p not in named]
    if 'distance' not in named and out['PULSAR'] is None:
        missing.append('distance')
    if missing:
        raise SourceError(
            f"{label}: {missing} must appear in PARAMETERS (to fit) or "
            f"FROZEN (to hold fixed).")
    return out


def source_settings(src) -> dict:
    """Return the ``OPTIONS``, ``FIT`` and ``NESTED`` blocks with defaults filled in."""
    blocks = _blocks(src)
    return {k: blocks[k] for k in ('OPTIONS', 'FIT', 'NESTED')}


# Building

def _parameter(model, name: str, block: str):
    if name not in model.parameters:
        raise SourceError(
            f"{block}: unknown parameter '{name}'. "
            f"Available: {model.parameters.names}.")
    return model.parameters[name]


def _physical(par, val, what: str) -> float:
    """Convert a value or prior width to the parameter's internal unit."""
    if par._unit is None:
        return float(val)
    return par._convert_quantity(val, context=what)


def _as_quantity(val, unit):
    return val if hasattr(val, 'unit') else val * unit


def build_from_source(src):
    """Build the model, observables and fit spec a source file describes.

    Parameters
    ----------
    src : module
        A loaded source file (see :func:`load_source`).

    Returns
    -------
    model : PWNModel
    observables : list of Observable
    spec : _FitSpec
    """
    import astropy.units as u

    blocks = _blocks(src)
    options = blocks['OPTIONS']

    want = options['constants']
    if want is not None and want != C.CONSTANTS_SOURCE:
        raise SourceError(
            f"OPTIONS['constants'] is '{want}' but viper loaded the "
            f"'{C.CONSTANTS_SOURCE}' set. Launch with "
            f"VIPER_CONSTANTS={want}.")

    model = PWNModel()
    if blocks['PULSAR'] is not None:
        model.set_pulsar(blocks['PULSAR'])
    if blocks['PERIOD'] is not None or blocks['P_DOT'] is not None:
        model.set_pulsar(period=blocks['PERIOD'], p_dot=blocks['P_DOT'])

    for name, entry in blocks['PARAMETERS'].items():
        par = _parameter(model, name, 'PARAMETERS')
        if not isinstance(entry, (list, tuple)) or len(entry) not in (3, 4):
            raise SourceError(
                f"PARAMETERS['{name}'] must be (start, low, high), "
                f"optionally followed by 'log' or 'linear'.")
        start, low, high = entry[:3]
        if len(entry) == 4:
            if entry[3] not in ('log', 'linear'):
                raise SourceError(
                    f"PARAMETERS['{name}']: the fourth item must be "
                    f"'log' or 'linear', got {entry[3]!r}.")
            par.log = entry[3] == 'log'
        par.value = start
        par.bounds = (low, high)
        par.frozen = False
        lo, hi = par.bounds
        if not lo <= par.value <= hi:
            raise SourceError(
                f"PARAMETERS['{name}']: start value {start} is outside "
                f"its bounds ({low}, {high}).")

    for name, value in blocks['FROZEN'].items():
        par = _parameter(model, name, 'FROZEN')
        par.value = value
        par.frozen = True

    for i, field in enumerate(blocks['IC_FIELDS'], start=1):
        unknown = sorted(set(field) - set(_IC_KEYS))
        if unknown or 'temp' not in field or 'norm' not in field:
            raise SourceError(
                f"IC_FIELDS[{i}] needs 'temp' and 'norm'; optional keys "
                f"are 'temp_bounds' and 'norm_bounds'. Got {sorted(field)}.")
        idx = model.add_ic_field(temp=field['temp'], norm=field['norm'],
                                 frozen=True)
        for key, unit in (('temp', u.K), ('norm', u.eV / u.cm**3)):
            bounds = field.get(f'{key}_bounds')
            if bounds is None:
                continue
            par = model.parameters[f'ic_{key}_{idx}']
            par.bounds = tuple(_as_quantity(b, unit) for b in bounds)
            par.frozen = False

    for name, entry in blocks['PRIORS'].items():
        par = _parameter(model, name, 'PRIORS')
        if not isinstance(entry, (list, tuple)) or len(entry) not in (2, 3):
            raise SourceError(
                f"PRIORS['{name}'] must be (center, sigma), optionally "
                f"followed by 'log'.")
        if len(entry) == 3:
            if entry[2] != 'log':
                raise SourceError(
                    f"PRIORS['{name}']: the third item must be 'log', "
                    f"got {entry[2]!r}.")
            # log10 space: unitless
            par.prior = (float(entry[0]), float(entry[1]))
            par.prior_log = True
        else:
            par.prior = (_physical(par, entry[0], 'prior center'),
                         _physical(par, entry[1], 'prior sigma'))

    # must run before build_fit_spec
    model.apply_run_config(options)

    observables = list(blocks['OBSERVABLES'])
    spec = build_fit_spec(model, observables,
                          integration=options['integration'])
    return model, observables, spec


# Run folders

def new_run_dir(source_path: str, root: str = 'results') -> str:
    """Return the next ``<root>/<name>_runNN`` for a source file.

    Numbering continues after the highest existing run of that name,
    including folders that carry a suffix (``<name>_run07_test``).
    """
    stem = os.path.splitext(os.path.basename(source_path))[0]
    pattern = re.compile(re.escape(stem) + r'_run(\d+)')
    used = [0]
    if os.path.isdir(root):
        for entry in os.listdir(root):
            m = pattern.match(entry)
            if m:
                used.append(int(m.group(1)))
    return os.path.join(root, f'{stem}_run{max(used) + 1:02d}')


def snapshot_path(run_dir: str) -> str:
    """Path of the source-file copy inside a run folder."""
    return os.path.join(run_dir, SNAPSHOT_NAME)


def start_run(source_path: str, run_dir: str) -> str:
    """Create ``run_dir`` and copy the source file into it.

    Returns the path of the copy.
    """
    os.makedirs(run_dir, exist_ok=True)
    dest = snapshot_path(run_dir)
    if os.path.abspath(source_path) != os.path.abspath(dest):
        shutil.copyfile(source_path, dest)
    return dest


def load_run(run_dir: str):
    """Load the source-file copy of an existing run folder."""
    path = snapshot_path(run_dir)
    if not os.path.isfile(path):
        raise SourceError(
            f"{run_dir} has no {SNAPSHOT_NAME}; it was not created by "
            f"this runner.")
    return load_source(path)
