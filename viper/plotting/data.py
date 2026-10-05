"""SED data-file loading and per-dataset display conventions."""

from __future__ import annotations

import csv
import numpy as np

from ..observables import ObsType
from .. import constants as C
from .style import _eV_to_erg, _h_eV



_DATASET_COLORS = {
    'VLA': 'blue',
    'CXO': 'red',
    'Chandra': 'red',
    'Fermi': 'orange',
    'VERITAS': 'purple',
    'WCDA': 'limegreen',
    'KM2A': 'darkgreen',
    'LHAASO': 'green',
}


def _color_for_label(label: str, colors: dict | None = None) -> str:
    """Pick color based on keywords in the observable label."""
    if colors is None:
        colors = _DATASET_COLORS
    for key, color in colors.items():
        if key in label:
            return color
    return 'gray'


def _unweighted_chi2(predicted, config):
    """Compute unweighted chi-squared (w=1 for all observables)."""
    chi2 = 0.0
    for k, obs in enumerate(config.observables):
        if obs.range_lo is not None and obs.range_hi is not None:
            if predicted[k] < obs.range_lo:
                chi2 += ((predicted[k] - obs.range_lo) / obs.error) ** 2
            elif predicted[k] > obs.range_hi:
                chi2 += ((predicted[k] - obs.range_hi) / obs.error) ** 2
        elif obs.upper_limit:
            if predicted[k] > obs.value:
                chi2 += ((predicted[k] - obs.value) / obs.limit_sigma) ** 2
        elif obs.lower_limit:
            if predicted[k] < obs.value:
                chi2 += ((predicted[k] - obs.value) / obs.limit_sigma) ** 2
        else:
            if obs.error is not None and obs.error > 0:
                chi2 += ((predicted[k] - obs.value) / obs.error) ** 2
    return chi2


# ===================================================================
# Bowtie detection and drawing
# ===================================================================

def _find_bowties(observables):
    """Detect spectral_index + flux_density pairs that form bowties.

    Matches a spectral_index (energy range [lo, hi]) with a
    'photons/cm2/s/...' flux_density norm at a pivot inside it.

    Returns list of dicts with keys:
        E_lo_eV, E_hi_eV, E_piv_eV, norm, norm_err, gamma, gamma_err,
        flux_unit, label, color
    """
    indices = [(i, obs) for i, obs in enumerate(observables)
               if obs.obs_type == ObsType.SPECTRAL_INDEX]
    densities = [(i, obs) for i, obs in enumerate(observables)
                 if obs.obs_type == ObsType.FLUX_DENSITY
                 and not obs.upper_limit
                 and 'photons/cm2/s/' in obs._flux_unit]

    bowties = []
    used_density_indices = set()

    for si_idx, si_obs in indices:
        E_lo_eV = si_obs._freq_lo_hz * _h_eV
        E_hi_eV = si_obs._freq_hi_hz * _h_eV

        for fd_idx, fd_obs in densities:
            if fd_idx in used_density_indices:
                continue
            E_piv_eV = fd_obs._freq_hz * _h_eV
            if E_lo_eV <= E_piv_eV <= E_hi_eV:
                used_density_indices.add(fd_idx)
                bowties.append({
                    'E_lo_eV': E_lo_eV,
                    'E_hi_eV': E_hi_eV,
                    'E_piv_eV': E_piv_eV,
                    'norm': fd_obs.value,
                    'norm_err': fd_obs.error,
                    'gamma': si_obs.value,
                    'gamma_err': si_obs.error,
                    'flux_unit': fd_obs._flux_unit,
                    'label': _bowtie_label(si_obs, fd_obs),
                    'color': _color_for_label(si_obs.label + fd_obs.label),
                    'si_idx': si_idx,
                    'fd_idx': fd_idx,
                })
                break

    # spectral_index + band_flux over the same band
    bands = [(i, obs) for i, obs in enumerate(observables)
             if obs.obs_type == ObsType.BAND_FLUX
             and not (obs.upper_limit or obs.lower_limit)]
    used_si = {bt['si_idx'] for bt in bowties}
    used_bands = set()
    for si_idx, si_obs in indices:
        if si_idx in used_si:
            continue
        for bf_idx, bf_obs in bands:
            if bf_idx in used_bands:
                continue
            if (np.isclose(bf_obs._freq_lo_hz, si_obs._freq_lo_hz)
                    and np.isclose(bf_obs._freq_hi_hz, si_obs._freq_hi_hz)):
                used_bands.add(bf_idx)
                bowties.append({
                    'E_lo_eV': si_obs._freq_lo_hz * _h_eV,
                    'E_hi_eV': si_obs._freq_hi_hz * _h_eV,
                    'band_flux': bf_obs.value,
                    'band_flux_err': bf_obs.error,
                    'gamma': si_obs.value,
                    'gamma_err': si_obs.error,
                    'label': _bowtie_label(si_obs, bf_obs),
                    'color': _color_for_label(si_obs.label + bf_obs.label),
                    'si_idx': si_idx,
                    'fd_idx': bf_idx,
                })
                break

    return bowties


def _band_shape(E, lo, hi, gamma):
    """νFν(E) of a power law with unit energy flux in [lo, hi]."""
    k = 2.0 - gamma
    if abs(k) < 1e-9:
        return np.full_like(E, 1.0 / np.log(hi / lo))
    return k * E**k / (hi**k - lo**k)


def _bowtie_label(si_obs, fd_obs):
    """Make a label for the bowtie from the component observables."""
    for key in _DATASET_COLORS:
        if key in si_obs.label or key in fd_obs.label:
            return key
    return fd_obs.label.split('[')[0].strip()


def _draw_bowtie(ax, bt, n_points=100):
    """Draw a bowtie (butterfly) region on the SED plot.

    Computes νFν = E² × dN/dE for all four corners of
    (norm ± err) × (gamma ± err) and fills between min/max.
    """
    E = np.geomspace(bt['E_lo_eV'], bt['E_hi_eV'], n_points)
    if 'band_flux' in bt:
        lo, hi = bt['E_lo_eV'], bt['E_hi_eV']
        f, df = bt['band_flux'], bt['band_flux_err']
        g, dg = bt['gamma'], bt['gamma_err']
        curves = np.array([(f + sf * df) * _band_shape(E, lo, hi, g + sg * dg)
                           for sf in (-1, 1) for sg in (-1, 1)])
        color = bt['color']
        ax.plot(E, f * _band_shape(E, lo, hi, g), c=color, lw=1.5,
                label=bt['label'])
        ax.fill_between(E, curves.min(axis=0), curves.max(axis=0),
                        color=color, alpha=0.2)
        return
    E_piv = bt['E_piv_eV']
    norm = bt['norm']
    norm_err = bt['norm_err']
    gamma = bt['gamma']
    gamma_err = bt['gamma_err']

    flux_unit = bt['flux_unit']
    eu = flux_unit.split('/')[-1]  # 'keV', 'GeV', etc.
    erg_per_unit = _eV_to_erg
    for suffix, scale in (('keV', 1.0e3), ('MeV', 1.0e6),
                          ('GeV', 1.0e9), ('TeV', 1.0e12)):
        if eu == suffix:
            erg_per_unit = scale / C.eV_per_erg
            break

    def nufnu(E_eV, n, g):
        """νFν = E² × dN/dE [erg/s/cm²].

        dN/dE = n × (E/E_piv)^{-g}  [photons/cm²/s/energy_unit]
        νFν = (E_eV × eV_to_erg)² × dN/dE / erg_per_unit
        """
        E_erg = E_eV * _eV_to_erg
        return E_erg**2 * n * (E_eV / E_piv)**(-g) / erg_per_unit

    curves = np.array([
        nufnu(E, norm + norm_err, gamma + gamma_err),
        nufnu(E, norm + norm_err, gamma - gamma_err),
        nufnu(E, norm - norm_err, gamma + gamma_err),
        nufnu(E, norm - norm_err, gamma - gamma_err),
    ])

    best = nufnu(E, norm, gamma)
    lo = curves.min(axis=0)
    hi = curves.max(axis=0)

    color = bt['color']
    ax.plot(E, best, c=color, lw=1.5, label=bt['label'])
    ax.fill_between(E, lo, hi, color=color, alpha=0.2)


# ===================================================================
# Data file loading
# ===================================================================

def load_sed_data(data_file: str) -> dict[str, list]:
    """Load SED data from a CSV table.

    Expected header columns: ``dataset``, ``Energy_eV``,
    ``Unabsorbed_Flux_erg/s/cm2``, ``Unabsorbed_Flux_err_lo_erg/s/cm2``,
    ``Unabsorbed_Flux_err_hi_erg/s/cm2``, ``upper_limit_flag``.

    Returns dict: dataset_name → [(E_eV, flux, err_lo, err_hi, upper_limit), ...]
    """
    data = {}
    with open(data_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ds = row['dataset']
            if ds not in data:
                data[ds] = []
            data[ds].append((
                float(row['Energy_eV']),
                float(row['Unabsorbed_Flux_erg/s/cm2']),
                float(row['Unabsorbed_Flux_err_lo_erg/s/cm2']),
                float(row['Unabsorbed_Flux_err_hi_erg/s/cm2']),
                int(row['upper_limit_flag']),
            ))
    return data


def _plot_data_file(ax, data_file: str, skip_datasets=None, label_once=None):
    """Overlay SED data from a CSV file onto the axes.

    Parameters
    ----------
    skip_datasets : set of str, optional
        Dataset names to skip (e.g. {'CXO', 'VERITAS'}).
    label_once : dict, optional
        Tracks which labels have been used to avoid duplicates.
    """
    if label_once is None:
        label_once = {}
    if skip_datasets is None:
        skip_datasets = set()

    data = load_sed_data(data_file)

    _legend_map = {
        'LHAASO_WCDA': 'WCDA',
        'LHAASO_KM2A': 'KM2A',
    }

    for ds, rows in data.items():
        if ds in skip_datasets:
            continue

        E = np.array([r[0] for r in rows])
        F = np.array([r[1] for r in rows])
        lo = np.array([r[2] for r in rows])
        hi = np.array([r[3] for r in rows])
        ul = np.array([r[4] for r in rows])

        color = _color_for_label(ds)
        legend_name = _legend_map.get(ds, ds)

        if legend_name in label_once:
            label = ''
        else:
            label = legend_name
            label_once[legend_name] = True

        ul_mask = ul.astype(bool)
        det_mask = ~ul_mask

        if det_mask.any():
            ax.errorbar(E[det_mask], F[det_mask],
                        yerr=[lo[det_mask], hi[det_mask]],
                        c=color, ls='none', marker='o', ms=4,
                        capsize=2, label=label)
            if label:
                label = ''

        if ul_mask.any():
            ax.errorbar(E[ul_mask], F[ul_mask],
                        yerr=F[ul_mask] * 0.6, uplims=True,
                        c=color, ls='none', marker='o', ms=4,
                        label=label)
