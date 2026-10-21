# Fit a source

Describe the source in one Python file and hand it to the runner.
`examples/g54.py` in the repository is a complete example: copy it,
replace the values with your own, and run

```bash
python3 tools/run_mcmc.py my_source.py
```

The runner scripts live in the repository, not in the installed
package, so this route needs a clone (see {doc}`install`).

## The source file

A source file is plain Python.  It defines these names:

| Name | What it holds | Required |
|---|---|---|
| `PULSAR` | ATNF pulsar name; sets the period, period derivative and distance | yes, or `PERIOD` and `P_DOT` |
| `OBSERVABLES` | list of `Observable` | yes |
| `PARAMETERS` | parameters to fit, `name: (start, low, high)` | yes |
| `FROZEN` | parameters held fixed, `name: value` | for every model parameter not being fit |
| `PRIORS` | Gaussian priors, `name: (center, sigma)` | no |
| `AGE` | known age (`tau_sd` then follows from the braking index), or an allowed `(low, high)` age range with `tau_sd` and the braking index both free | no |
| `IC_FIELDS` | photon fields for inverse-Compton scattering, on top of the CMB | no |
| `OPTIONS` | model switches | no |
| `FIT` | MCMC settings | no |
| `NESTED` | nested-sampling settings | no |

Values with a physical dimension are astropy Quantities:

```python
import astropy.units as u
from viper import Observable

PULSAR = 'J1930+1852'

OBSERVABLES = [
    Observable.angular_size_pwn(1.14, 0.04, 'arcmin'),
    Observable.flux_density(1.4, 'GHz', 433, 30, 'mJy'),
    Observable.band_flux(2, 10, 'keV', 5.43e-12, 0.035e-12, weight=2),
]

PARAMETERS = {
    'mej':    (21 * u.M_sun, 1 * u.M_sun, 100 * u.M_sun),
    'tau_sd': (3574 * u.yr, 100 * u.yr, 1e5 * u.yr),
}

FROZEN = {
    'esn': 0.95e51 * u.erg,
    # ... every other model parameter
}
```

The eleven model parameters `esn`, `mej`, `n_ism`, `braking_index`,
`tau_sd`, `eta_b`, `e_min`, `e_max`, `e_break`, `p1` and `p2` must each
appear in `PARAMETERS` or `FROZEN`.  `distance` may be left out when
`PULSAR` is given; the catalogue value is then used.

### Observables

Every `Observable` constructor takes `weight=`, which multiplies that
point's chi-square term.  `flux_density` and `band_flux` take
`upper_limit=True`; `band_flux` also takes `lower_limit=True`.  Because
the file is Python, a table and a loop work too:

```python
XRAY = [
    # energy [keV], flux [erg/s/cm2], error
    (1.18, 4.11e-13, 7.4e-14),
    (1.46, 5.16e-13, 6.2e-14),
]
OBSERVABLES += [
    Observable.flux_density(e, 'keV', f, err, 'erg/s/cm2',
                            weight=3 if e > 5 else 1)
    for e, f, err in XRAY
]
```

### Parameters

`esn`, `mej`, `n_ism`, `tau_sd`, `eta_b`, `e_min`, `e_max` and `e_break`
are sampled in log10 space; `braking_index`, `p1`, `p2` and `distance`
linearly.  Add `'log'` or `'linear'` as a
fourth item to change that for one parameter:

```python
'e_break': (3232 * u.GeV, 10 * u.GeV, 1e5 * u.GeV, 'linear'),
```

A prior is Gaussian in the parameter's own units.  Add `'log'` as a
third item for a prior in log10 space:

```python
PRIORS = {
    'distance': (6.2 * u.kpc, 1.0 * u.kpc),
    'eta_b':    (-2.0, 1.0, 'log'),
}
```

### Photon fields

```python
IC_FIELDS = [
    {'temp': 35 * u.K, 'norm': 0.35 * u.eV / u.cm**3},          # held fixed
    {'temp': 3000 * u.K, 'norm': 0.5 * u.eV / u.cm**3,
     'norm_bounds': (0.1 * u.eV / u.cm**3, 5 * u.eV / u.cm**3)},  # norm is fit
]
```

`norm` is the energy density of the field.  A temperature or energy
density is fit when it has bounds, and held fixed otherwise.

### Settings

| Block | Key | Default |
|---|---|---|
| `FIT` | `nwalkers` | 96 |
| | `nsteps` | 1000 |
| | `scatter` (walker spread around the start, as a fraction) | 0.005 |
| | `seed` | 42 |
| | `init_from` (a `best_theta.npy` to start from) | none |
| `NESTED` | `nlive` | 800 |
| | `dlogz` | 0.5 |
| | `slice_steps` | `max(2 * ndim, 8)` |
| | `resume` | `'resume'` |
| `OPTIONS` | `integration` | `'pc_full'` |
| | `constants` (refuse to run under the other constant set) | not checked |
| | `use_cmb_lookup` | `True` |
| | `snapshot_mode` | `'secant'` |
| | `timestep_events` | `False` |
| | `event_dt_floor_yr` | 0.5 |
| | `timestep_cooling` | `'compat'` |
| | `resolve_final_cooling` (resolve fast cooling before the final spectrum) | `False` |
| | `escape_model` | `'off'` |
| | `injection_model` | `'broken_powerlaw'` |
| | `ssc_model` | `'off'` |

A misspelt key or block name stops the run with a message naming it.

## Run folders

Each run gets its own folder: `results/my_source_run01`, then
`_run02`, and so on (`--outdir` picks another place).

| File | Content |
|---|---|
| `source.py` | copy of the source file as it was when the run started |
| `chain.h5` | the emcee chain |
| `best_theta.npy` | best point of the chain, in fit space |
| `mcmc.pid` | process id while the run is active |

Extending or inspecting a run reads the copy in its folder, so editing
the original file later never changes what an earlier run means.  For
the same reason, keep the data in the source file itself; a file read
through a relative path would not be found from the copy.

```bash
python3 tools/run_mcmc.py results/my_source_run01 --extend 2000
python3 tools/monitor.py results/my_source_run01/chain.h5
./tools/stop_mcmc.sh results/my_source_run01
```

## Nested sampling

```bash
python3 tools/run_nested.py my_source.py --ncores 4
```

The same source file works for both samplers.  Re-running the run
folder resumes an interrupted run.

## From Python

```python
from viper import load_source, build_from_source, run_mcmc

model, observables, spec = build_from_source(load_source('my_source.py'))
sampler, spec = run_mcmc(model, observables, nwalkers=32, nsteps=2000)
```
