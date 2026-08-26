"""
CTA 1 (PSR J0007+7303) configuration for MCMC fitting.

Usage:
    from examples.cta1_config import build_cta1_model, build_cta1_observables
    from viper.fitting import build_fit_spec

    model = build_cta1_model()
    obs   = build_cta1_observables()
    spec  = build_fit_spec(model, obs)
"""
import astropy.units as u
from dataclasses import replace
from viper import PWNModel, Observable


# ----------------------------------------------------------------
# Observational data
# ----------------------------------------------------------------

# Chandra (CXO) spectral points (unabsorbed nuFnu)
CXO_DATA = [
    # (energy_eV, nuFnu_erg_s_cm2, error)
    (1175.3, 4.1081e-13, 7.35e-14),
    (1460.0, 5.16363e-13, 6.19e-14),
    (1693.6, 5.23964e-13, 6.13e-14),
    (1978.3, 5.56428e-13, 6.63e-14),
    (2365.2, 5.72504e-13, 8.33e-14),
    (2825.1, 6.06943e-13, 8.30e-14),
    (3358.0, 5.64628e-13, 8.30e-14),
    (3993.1, 5.13238e-13, 9.28e-14),
    (4766.9, 4.89687e-13, 1.14e-13),
    (5708.6, 7.3215e-13,  1.87e-13),
]

# Fermi-LAT GWU analysis (energy_eV, nuFnu, err_hi, err_lo)
FERMI_DATA = [
    (6.746414e10, 3.4974e-12, 8.9647e-13, 8.9647e-13),
    (1.228228e11, 2.6860e-12, 1.1252e-12, 1.1252e-12),
    (2.236068e11, 1.7521e-12, 1.6577e-12, 1.6577e-12),
    (4.070905e11, 3.0195e-12, 1.7317e-12, 1.7317e-12),
    (7.411344e11, 5.8930e-12, 3.4781e-12, 3.4781e-12),
]

# LHAASO WCDA (energy_eV, nuFnu, err_hi, err_lo)
WCDA_DATA = [
    (7.5e12,  3.46e-12, 5.60e-13, 5.90e-13),
    (1.18e13, 2.95e-12, 4.70e-13, 4.50e-13),
    (1.72e13, 2.44e-12, 3.60e-13, 4.20e-13),
    (3.05e13, 2.15e-12, 3.50e-13, 4.10e-13),
    (5.73e13, 1.23e-12, 3.03e-13, 3.30e-13),
]

# LHAASO KM2A (energy_eV, nuFnu, err_hi, err_lo)
KM2A_DATA = [
    (4.87e13, 1.27e-12, 2.00e-13, 2.20e-13),
    (7.91e13, 9.72e-13, 1.02e-13, 9.80e-14),
    (1.09e14, 5.67e-13, 6.70e-14, 7.70e-14),
    (1.75e14, 2.78e-13, 5.20e-14, 6.90e-14),
    (2.50e14, 1.09e-13, 3.90e-14, 5.60e-14),
]

# VERITAS (energy_eV, nuFnu, err_hi, err_lo) -- not used in current fits
VERITAS_DATA = [
    (7.41e11, 1.68e-12, 7.56e-13, 7.60e-13),
    (1.29e12, 2.05e-12, 4.60e-13, 4.60e-13),
    (2.25e12, 1.07e-12, 3.55e-13, 3.50e-13),
    (4.19e12, 9.51e-13, 3.13e-13, 3.09e-13),
    (7.30e12, 1.07e-12, 4.50e-13, 4.70e-13),
    (1.36e13, 1.94e-12, 6.80e-13, 7.10e-13),
]


# ----------------------------------------------------------------
# Observable builder
# ----------------------------------------------------------------

def build_cta1_observables(
    cxo_weight=1.0,
    cxo_scale=1.0,
    include_veritas=False,
    xray_band_limits=None,
):
    """Build the standard CTA 1 observable list.

    Parameters
    ----------
    cxo_weight : float, weight for CXO points (default 1.0)
    cxo_scale : float, scale on CXO flux and error (default 1.0)
    xray_band_limits : dict, band-flux limits replacing the CXO points; keys lo_keV, hi_keV, lower_limit, lower_limit_err, upper_limit, upper_limit_err
    include_veritas : bool, include VERITAS data (default False)

    Returns
    -------
    list of Observable
    """
    obs = []

    # Sizes
    obs.append(Observable.angular_size_snr(45, 10, 'arcmin'))
    obs.append(Observable.angular_size_pwn_range(10.2, 13.8, 1.8, 'arcmin'))

    # Radio upper limit
    obs.append(Observable.flux_density(
        5.97e-6, 'eV', 1.24e-15, 0.0, 'erg/s/cm2',
        label='VLA 1.4 GHz [UL]', upper_limit=True))

    # CXO
    if xray_band_limits is not None:
        bl = xray_band_limits
        obs.append(Observable.band_flux(
            bl['lo_keV'], bl['hi_keV'], 'keV',
            bl['lower_limit'], bl['lower_limit_err'],
            label=f'CXO {bl["lo_keV"]}-{bl["hi_keV"]} keV [LL]',
            lower_limit=True))
        obs.append(Observable.band_flux(
            bl['lo_keV'], bl['hi_keV'], 'keV',
            bl['upper_limit'], bl['upper_limit_err'],
            label=f'ASCA {bl["lo_keV"]}-{bl["hi_keV"]} keV [UL]',
            upper_limit=True))
    else:
        for e, f, err in CXO_DATA:
            ob = Observable.flux_density(
                e, 'eV', f * cxo_scale, err * cxo_scale, 'erg/s/cm2',
                label=f'CXO {e/1e3:.2f} keV')
            if cxo_weight != 1.0:
                ob = replace(ob, weight=cxo_weight)
            obs.append(ob)

    # Fermi
    for e, f, ehi, elo in FERMI_DATA:
        obs.append(Observable.flux_density(
            e, 'eV', f, 0.5*(ehi+elo), 'erg/s/cm2',
            label=f'Fermi {e/1e9:.0f} GeV'))

    # VERITAS
    if include_veritas:
        for e, f, ehi, elo in VERITAS_DATA:
            obs.append(Observable.flux_density(
                e, 'eV', f, 0.5*(ehi+elo), 'erg/s/cm2',
                label=f'VERITAS {e/1e12:.1f} TeV'))

    # WCDA
    for e, f, ehi, elo in WCDA_DATA:
        obs.append(Observable.flux_density(
            e, 'eV', f, 0.5*(ehi+elo), 'erg/s/cm2',
            label=f'WCDA {e/1e12:.1f} TeV'))

    # KM2A
    for e, f, ehi, elo in KM2A_DATA:
        obs.append(Observable.flux_density(
            e, 'eV', f, 0.5*(ehi+elo), 'erg/s/cm2',
            label=f'KM2A {e/1e12:.0f} TeV'))

    return obs


# ----------------------------------------------------------------
# Model builder
# ----------------------------------------------------------------

# Default parameter bounds
DEFAULT_BOUNDS = {
    'esn':           (1e49 * u.erg,           3e52 * u.erg),
    'mej':           (0.5 * u.M_sun,          100 * u.M_sun),
    'n_ism':         (1e-3 * u.Unit('1/cm3'), 10 * u.Unit('1/cm3')),
    'distance':      (400 * u.pc,             3000 * u.pc),
    'braking_index': (1.5,                    3.5),
    'tau_sd':        (100 * u.yr,             1e5 * u.yr),
    'eta_b':         (1e-4,                   0.5),
    'e_min':         (0.1 * u.GeV,            1000 * u.GeV),
    'e_max':         (1e4 * u.GeV,            1e7 * u.GeV),
    'e_break':       (1 * u.GeV,              1e5 * u.GeV),
    'p1':            (0.5,                    2.5),
    'p2':            (1.5,                    3.5),
}

# Default priors: (center, sigma) in physical units
DEFAULT_PRIORS = {
    'distance': (1400, 400),
}

RUN_34_PARAMS = {
    'esn':           1.0e51 * u.erg,
    'mej':           9.8 * u.M_sun,
    'n_ism':         0.15 * u.Unit("1/cm3"),
    'braking_index': 3.0,
    'tau_sd':        5000 * u.yr,
    'eta_b':         0.06,
    'e_min':         1.1 * u.GeV,
    'e_max':         263e3 * u.GeV,
    'e_break':       10000 * u.GeV,
    'p1':            2.5,
    'p2':            1.6,
    'distance':      1400 * u.pc,
}

RUN_35_PARAMS = {
    'esn':           1.0e51 * u.erg,
    'mej':           9.8 * u.M_sun,
    'n_ism':         1.0 * u.Unit("1/cm3"),
    'braking_index': 2.6,
    'tau_sd':        400 * u.yr,
    'eta_b':         0.001,
    'e_min':         1.1 * u.GeV,
    'e_max':         263e3 * u.GeV,
    'e_break':       1000 * u.GeV,
    'p1':            2.5,
    'p2':            1.6,
    'distance':      1400 * u.pc,
}


LOG_PARAMS = {'esn', 'mej', 'n_ism', 'tau_sd', 'eta_b',
              'e_min', 'e_max', 'e_break'}


def build_cta1_model(
    params=None,
    priors=None,
    bounds=None,
    ic_fields=None,
):
    """Build a PWNModel for CTA 1.

    Parameters
    ----------
    params : dict, starting values in physical units (None = model defaults)
    priors : dict, Gaussian priors {name: (center, sigma)} (default DEFAULT_PRIORS)
    bounds : dict, parameter bounds (default DEFAULT_BOUNDS)
    ic_fields : list of dict, extra IC fields; keys temp (K), norm, frozen, temp_bounds, norm_bounds

    Returns
    -------
    PWNModel
    """
    if priors is None:
        priors = DEFAULT_PRIORS
    if bounds is None:
        bounds = DEFAULT_BOUNDS

    model = PWNModel()
    model.set_pulsar('J0007+7303')

    for name in DEFAULT_BOUNDS:
        p = getattr(model, name)
        if params is not None and name in params:
            p.value = params[name]
        p.frozen = False
        p.log = name in LOG_PARAMS
        p.bounds = bounds[name]
        if name in priors:
            p.prior = priors[name]

    if ic_fields:
        for ic in ic_fields:
            idx = model.add_ic_field(
                temp=ic['temp'],
                norm=ic['norm'],
                frozen=ic.get('frozen', False),
            )
            if 'temp_bounds' in ic:
                model.parameters[f'ic_temp_{idx}'].bounds = ic['temp_bounds']
            if 'norm_bounds' in ic:
                model.parameters[f'ic_norm_{idx}'].bounds = ic['norm_bounds']

    return model


# ----------------------------------------------------------------
# Config-driven builder (called by run_mcmc.py and monitor.py)
# ----------------------------------------------------------------

def build_from_config(run_cfg):
    """Build model, observables, and fit spec from a run_config dict.

    Returns
    -------
    model, obs, spec
    """
    import astropy.units as u_cfg
    from viper.fitting import build_fit_spec

    cxo_weight = run_cfg.get('cxo_weight', 2.0)
    cxo_scale = run_cfg.get('cxo_scale', 1.0)
    xray_band_limits = run_cfg.get('xray_band_limits', None)
    frozen_params = run_cfg.get('frozen_params', {})
    mej_prior = run_cfg.get('mej_prior', None)
    emax_bounds = run_cfg.get('e_max_bounds', None)
    ebreak_bounds = run_cfg.get('e_break_bounds', None)

    dist_prior_val = run_cfg.get('distance_prior', [1400, 400])
    priors = {'distance': tuple(dist_prior_val)}
    if mej_prior is not None:
        priors['mej'] = tuple(mej_prior)
    custom_bounds = None
    if emax_bounds or ebreak_bounds:
        custom_bounds = dict(DEFAULT_BOUNDS)
        if emax_bounds:
            custom_bounds['e_max'] = (10**emax_bounds[0] * u_cfg.GeV,
                                      10**emax_bounds[1] * u_cfg.GeV)
        if ebreak_bounds:
            custom_bounds['e_break'] = (10**ebreak_bounds[0] * u_cfg.GeV,
                                        10**ebreak_bounds[1] * u_cfg.GeV)

    ic_fields_cfg = run_cfg.get('ic_fields', None)
    if ic_fields_cfg:
        for ic in ic_fields_cfg:
            if 'temp_bounds' in ic:
                ic['temp_bounds'] = tuple(t * u_cfg.K for t in ic['temp_bounds'])
            if 'norm_bounds' in ic:
                ic['norm_bounds'] = tuple(ic['norm_bounds'])

    params_name = run_cfg.get('params', None)
    params = None
    if params_name:
        import examples.cta1_config as _self
        params = getattr(_self, params_name, None)
        if params is None:
            raise ValueError(
                f"params dict '{params_name}' not found in "
                f"examples/cta1_config.py")

    model = build_cta1_model(params=params, priors=priors,
                             bounds=custom_bounds, ic_fields=ic_fields_cfg)
    # frozen_params values are internal units
    model.apply_frozen_params(frozen_params)
    for pname in frozen_params:
        print(f"Frozen {pname} = {model.parameters[pname].value}")
    obs = build_cta1_observables(cxo_weight=cxo_weight, cxo_scale=cxo_scale,
                                   xray_band_limits=xray_band_limits)

    # must run before build_fit_spec
    model.apply_run_config(run_cfg)
    if (model.timestep_events or model.snapshot_mode != 'secant'
            or model.timestep_cooling != 'compat'):
        print(f"stepping options: timestep_events={model.timestep_events} "
              f"snapshot_mode={model.snapshot_mode} "
              f"timestep_cooling={model.timestep_cooling} "
              f"event_dt_floor_yr={model.event_dt_floor_yr}")
    if model.escape_model != 'off':
        print(f"escape: model={model.escape_model} "
              f"esc_norm={model.parameters['esc_norm'].value} "
              f"esc_index={model.parameters['esc_index'].value}")
    if model.injection_model != 'broken_powerlaw':
        print(f"injection: model={model.injection_model} "
              f"(e_break/p1/p2 reinterpreted per form; injection.py)")

    spec = build_fit_spec(model, obs, integration='pc_full')
    print(f"{spec.ndim} free params: {spec.param_names}")
    return model, obs, spec
