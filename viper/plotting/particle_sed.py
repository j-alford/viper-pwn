"""Combined SED + particle-spectrum figure."""

from __future__ import annotations

import numpy as np

from ..observables import ObsType
from .. import constants as C
from .style import (
    _particle_bin_edges,
    _FADED_STEP_DEFAULT_DEX,
    _align_ylabels_left,
    _apply_sed_style,
    _eV_to_erg,
    _h_eV,
    _plot_binned,
    _validate_bin_style,
)
from .data import _DATASET_COLORS, _color_for_label, _plot_data_file




# ===================================================================
# Combined SED + particle spectrum plot
# ===================================================================

# Insertion order sets keyword-lookup precedence.
_PARTICLE_SED_DATA_COLORS = {
    **_DATASET_COLORS,
    'Fermi': 'darkorange',
    'WCDA': 'green',
    'ASCA': 'purple',
}


def _particle_sed_color(label):
    return _color_for_label(label, _PARTICLE_SED_DATA_COLORS)


def _particle_sed_dataset_key(label):
    for key in _PARTICLE_SED_DATA_COLORS:
        if key in label:
            return key
    return label.split('[')[0].strip()


def _band_e_from_sync(E_photon_eV, B_uG):
    """Electron energy [eV] whose synchrotron ν_peak ≈ E_photon at B_µG.

    Uses E_photon ≈ 1.93e-8 * B_µG * E_GeV² eV (the ν_peak convention,
    i.e. h * 1.22e6 * γ² * B_G).
    """
    E_GeV = np.sqrt(E_photon_eV / (1.93e-8 * B_uG))
    return E_GeV * 1e9


def _band_e_from_ic_cmb(E_photon_eV):
    """Electron energy [eV] whose Thomson IC on CMB peaks at E_photon.

    Uses E_photon ≈ (4/3) γ² ε_CMB with ε_CMB ≈ 6.3e-4 eV
    (mean CMB photon energy at T = 2.73 K).
    """
    E_GeV = 0.000511 * np.sqrt(E_photon_eV / ((4.0 / 3.0) * 6.3e-4))
    return E_GeV * 1e9


def plot_sed_with_particles(config, theta, ax=None, figsize=(10, 7),
                            data_file=None, skip_datasets=None,
                            show_residuals=True,
                            model_label='Model',
                            particle_ylim=(1e42, 2e47),
                            photon_xlim=(1e-6, 1e15),
                            photon_ylim=(1e-16, 1e-10),
                            particle_xlim=None,
                            color_by_band=True,
                            phot_bin_style='curve',
                            elec_bin_style='faded_step',
                            fade_dex=_FADED_STEP_DEFAULT_DEX,
                            apply_style=True):
    """SED plot with overlaid particle spectrum.

    Left y-axis: νFν [erg cm⁻² s⁻¹] (model SED + data)
    Right y-axis: E² dN/dE [erg] (evolved particle spectrum)
    Bottom x-axis: Photon Energy [eV]
    Top x-axis: Particle Energy [eV]

    Top-axis decade ticks align with bottom-axis gridlines
    (1 top decade = 3 bottom); centered on the particle spectrum.

    If color_by_band=True, the particle spectrum is color-coded by
    contributing photon band (synchrotron and Thomson-IC-on-CMB
    mappings from the model B field); gray dashed = full spectrum.

    Parameters
    ----------
    config : FitSpec
    theta : array-like
    ax : matplotlib axes, optional
        Existing axes; ignored when ``show_residuals=True``.
    figsize : tuple
    data_file : str, optional
        SED data CSV; ``FLUX_DENSITY`` observables from ``config`` are
        then skipped to avoid duplicates.
    skip_datasets : set, optional
        Dataset names to skip from ``data_file``.
    show_residuals : bool
    model_label : str
    particle_ylim : tuple
    photon_xlim : tuple
    photon_ylim : tuple
    particle_xlim : tuple, optional (auto-computed if None)
    color_by_band : bool
    phot_bin_style : {'curve', 'step', 'shaded_step', 'faded_step'}, default 'curve'
        Visual representation of the photon SED.  See :func:`plot_sed`.
    elec_bin_style : {'curve', 'step', 'shaded_step', 'faded_step'}, default 'faded_step'
        Visual representation of the particle spectrum.  Band-coloring
        is preserved across all four styles.
    fade_dex : float, default 1.0
        Number of decades over which the 'faded_step' fill reaches zero.

    Returns
    -------
    fig : matplotlib Figure
    axes : dict
        Keys: ``'sed'``, ``'particles'``, ``'residuals'``, ``'top_axis'``.
        ``'particles'`` and ``'top_axis'`` are the same axes;
        ``'residuals'``/``'particles'`` are None when absent.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.gridspec import GridSpec
    from ..fitting import predict_observables

    _validate_bin_style(phot_bin_style)
    _validate_bin_style(elec_bin_style)

    if apply_style:
        _apply_sed_style()

    predicted, result = predict_observables(theta, config)

    distance = config.all_values.get('distance', 1000)
    for i, fp in enumerate(config.free_params):
        if fp.name == 'distance':
            distance = theta[i] if not fp.log else 10 ** theta[i]
            break
    d_cm = distance * C.pc

    freq = result.photon_freqs
    phot_spec = result.photon_spectrum
    phot_mask = phot_spec > 0
    energy_eV = freq * _h_eV
    nufnu = 1e40 * (phot_spec * freq) / (4 * np.pi * d_cm ** 2)

    # --- Figure / axes ---
    if show_residuals:
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05, figure=fig)
        ax_sed = fig.add_subplot(gs[0])
        ax_res = fig.add_subplot(gs[1], sharex=ax_sed)
        plt.setp(ax_sed.get_xticklabels(), visible=False)
    elif ax is None:
        fig, ax_sed = plt.subplots(figsize=figsize)
        ax_res = None
    else:
        ax_sed = ax
        fig = ax_sed.get_figure()
        ax_res = None

    # --- Model SED curve ---
    phot_bin_min_eV = result.phot_grid.bin_min * _h_eV
    phot_bin_max_eV = result.phot_grid.bin_max * _h_eV
    _plot_binned(
        ax_sed, phot_bin_style,
        energy_eV[phot_mask], nufnu[phot_mask],
        phot_bin_min_eV[phot_mask], phot_bin_max_eV[phot_mask],
        fade_dex=fade_dex,
        color='k', ls='-', lw=1.5,
        label=model_label, zorder=5,
    )

    # --- Data points: from file if given, otherwise from observables ---
    label_once = {}
    if data_file is not None:
        _plot_data_file(ax_sed, data_file, skip_datasets=skip_datasets,
                        label_once=label_once)

    # BAND_FLUX limits come only from observables.
    _trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    for k, obs in enumerate(config.observables):
        if obs.obs_type in (ObsType.ANGULAR_SIZE_SNR,
                            ObsType.ANGULAR_SIZE_PWN,
                            ObsType.SPECTRAL_INDEX):
            continue

        if obs.obs_type == ObsType.FLUX_DENSITY:
            if data_file is not None:
                continue
            if obs._flux_unit not in ('erg/s/cm2', 'nufnu'):
                continue
            E_eV = obs._freq_hz * _h_eV
            color = _particle_sed_color(obs.label)
            ds_key = _particle_sed_dataset_key(obs.label)
            label = '' if ds_key in label_once else ds_key
            if label:
                label_once[ds_key] = True
            if obs.upper_limit:
                ax_sed.errorbar(E_eV, obs.value, yerr=obs.value * 0.6,
                                uplims=True, c=color, ls='none',
                                marker='o', ms=4, label=label)
            else:
                ax_sed.errorbar(E_eV, obs.value, yerr=obs.error,
                                c=color, ls='none', marker='o', ms=4,
                                capsize=2, label=label)

        elif obs.obs_type == ObsType.BAND_FLUX:
            freq_lo = obs._freq_lo_hz
            freq_hi = obs._freq_hi_hz
            band_mask = ((freq >= freq_lo) & (freq <= freq_hi)
                         & (phot_spec > 0))
            if band_mask.sum() < 2:
                continue
            band_freq = freq[band_mask]
            band_nufnu = nufnu[band_mask]
            band_fnu = (1e40 * phot_spec[band_mask]
                        / (4 * np.pi * d_cm ** 2))
            model_band_flux = _trapz(band_fnu, band_freq)
            scale = (obs.value / model_band_flux
                     if model_band_flux > 0 else 1.0)
            band_e = band_freq * _h_eV
            scaled_nufnu = band_nufnu * scale
            color = _particle_sed_color(obs.label)
            ds_key = _particle_sed_dataset_key(obs.label)
            label = '' if ds_key in label_once else ds_key
            if label:
                label_once[ds_key] = True
            mid = len(band_e) // 2
            if obs.lower_limit:
                ax_sed.plot(band_e, scaled_nufnu, color=color, ls='-',
                            lw=1.5,
                            label=f'{label} (LL)' if label else '')
                ax_sed.annotate('',
                                xy=(band_e[mid], scaled_nufnu[mid] * 2),
                                xytext=(band_e[mid], scaled_nufnu[mid]),
                                arrowprops=dict(arrowstyle='->',
                                                color=color, lw=1.5))
            elif obs.upper_limit:
                ax_sed.plot(band_e, scaled_nufnu, color=color, ls='--',
                            lw=1.5,
                            label=f'{label} (UL)' if label else '')
                ax_sed.annotate('',
                                xy=(band_e[mid], scaled_nufnu[mid] / 2),
                                xytext=(band_e[mid], scaled_nufnu[mid]),
                                arrowprops=dict(arrowstyle='->',
                                                color=color, lw=1.5))

    # --- Particle spectrum overlay ---
    ax_overlay = None
    if result.elec_grid is not None:
        grid = result.elec_grid
        interior = slice(1, grid.n_bins - 1)
        e_gev = grid.bin_avg[interior]
        e_ev = e_gev * 1e9
        e_erg = e_ev * _eV_to_erg
        _pw, _pbmax = _particle_bin_edges(grid, interior,
                                          getattr(result, 'e_max_inj', None))
        widths_ev = _pw * 1e9
        elec_spec = result.electron_spectrum[interior]
        emask = elec_spec > 0

        if emask.any():
            dnde = elec_spec[emask] * 1e40 / widths_ev[emask]
            e2dnde = (e_erg[emask]) ** 2 * dnde * (1.0 / _eV_to_erg)
            x_all = e_ev[emask]
            y_all = e2dnde
            elec_bin_min_eV = grid.bin_min[interior] * 1e9
            elec_bin_max_eV = _pbmax * 1e9
            x_bin_min = elec_bin_min_eV[emask]
            x_bin_max = elec_bin_max_eV[emask]

            pos = ax_sed.get_position()
            ax_overlay = fig.add_axes(pos, frameon=False)
            ax_overlay.set_xscale('log')
            ax_overlay.set_yscale('log')

            # --- Top axis: 1/3 of bottom decades, integer-aligned ---
            bottom_decades = (np.log10(photon_xlim[1])
                              - np.log10(photon_xlim[0]))
            top_decades = bottom_decades / 3.0

            if particle_xlim is None:
                above_floor = y_all >= particle_ylim[0]
                if above_floor.any():
                    log_lo_data = np.log10(x_all[above_floor].min())
                    log_hi_data = np.log10(x_all[above_floor].max())
                else:
                    log_lo_data = np.log10(x_all.min())
                    log_hi_data = np.log10(x_all.max())
                log_center = 0.5 * (log_lo_data + log_hi_data)
                log_top_lo = int(round(log_center - top_decades / 2))
                log_top_hi = log_top_lo + top_decades
                particle_xlim_used = (10.0 ** log_top_lo,
                                      10.0 ** log_top_hi)
            else:
                particle_xlim_used = particle_xlim
                log_top_lo = np.log10(particle_xlim_used[0])
                log_top_hi = np.log10(particle_xlim_used[1])

            ax_overlay.set_xlim(particle_xlim_used)
            ax_overlay.set_ylim(particle_ylim)

            # --- Color by photon band ---
            bpwn_uG = None
            if color_by_band and result.dyninfo:
                bpwn_uG = getattr(result.dyninfo[-1], 'bpwn', None)

            if color_by_band and bpwn_uG and bpwn_uG > 0:
                band_defs = [
                    ('VLA',   5.97e-6 / 2.0, 5.97e-6 * 2.0,
                     'blue',       r'radio $e^{\pm}$',  'sync'),
                    ('CXO',   500.0,         7000.0,
                     'red',        r'X-ray $e^{\pm}$',  'sync'),
                    ('Fermi', 67e9,          741e9,
                     'darkorange', r'Fermi $e^{\pm}$',  'ic'),
                    ('WCDA',  7.5e12,        57e12,
                     'green',      r'WCDA $e^{\pm}$',   'ic'),
                    ('KM2A',  49e12,         250e12,
                     'darkgreen',  r'KM2A $e^{\pm}$',   'ic'),
                ]
                bands = []
                for nm, plo, phi, color, lab, mapping in band_defs:
                    if mapping == 'sync':
                        e_lo = _band_e_from_sync(plo, bpwn_uG)
                        e_hi = _band_e_from_sync(phi, bpwn_uG)
                    else:
                        e_lo = _band_e_from_ic_cmb(plo)
                        e_hi = _band_e_from_ic_cmb(phi)
                    bands.append((nm, min(e_lo, e_hi), max(e_lo, e_hi),
                                  color, lab))

                _plot_binned(
                    ax_overlay, elec_bin_style,
                    x_all, y_all, x_bin_min, x_bin_max,
                    fade_dex=fade_dex,
                    color='gray', ls='--',
                    lw=0.8, alpha=0.5, zorder=2,
                )

                # band-overlap bins stay gray
                band_count = np.zeros(len(x_all), dtype=int)
                for _, lo, hi, _, _ in bands:
                    band_count += ((x_all >= lo) & (x_all <= hi)).astype(int)

                for _, lo, hi, color, lab in bands:
                    seg_mask = ((x_all >= lo) & (x_all <= hi)
                                & (band_count == 1))
                    if seg_mask.sum() < 2:
                        continue
                    idx = np.where(seg_mask)[0]
                    breaks = np.where(np.diff(idx) > 1)[0]
                    starts = np.concatenate([[0], breaks + 1])
                    ends = np.concatenate([breaks, [len(idx) - 1]])
                    first = True
                    for s, e_ in zip(starts, ends):
                        seg = idx[s:e_ + 1]
                        _plot_binned(
                            ax_overlay, elec_bin_style,
                            x_all[seg], y_all[seg],
                            x_bin_min[seg], x_bin_max[seg],
                            fade_dex=fade_dex,
                            color=color, lw=2,
                            label=lab if first else None,
                            zorder=3,
                        )
                        first = False
            else:
                _plot_binned(
                    ax_overlay, elec_bin_style,
                    x_all, y_all, x_bin_min, x_bin_max,
                    fade_dex=fade_dex,
                    color='blue', ls='--',
                    lw=1.2, label=r'$e^{\pm}$', zorder=3,
                )

            # --- Top x-axis ticks at integer decades ---
            ax_overlay.xaxis.set_label_position('top')
            ax_overlay.xaxis.tick_top()
            ax_overlay.set_xlabel('Particle Energy [eV]', labelpad=12)
            tick_decades = np.arange(int(np.ceil(log_top_lo)),
                                     int(np.floor(log_top_hi)) + 1)
            ax_overlay.xaxis.set_major_locator(
                ticker.FixedLocator([10.0 ** k for k in tick_decades]))
            ax_overlay.xaxis.set_major_formatter(
                ticker.LogFormatterSciNotation())
            ax_overlay.xaxis.set_minor_locator(ticker.NullLocator())

            # --- Right y-axis labels for the particle spectrum ---
            ax_overlay.yaxis.set_label_position('right')
            ax_overlay.yaxis.tick_right()
            ax_overlay.set_ylabel(r'$E^2\,dN/dE$ [erg]', labelpad=8)

    # --- Residuals ---
    if ax_res is not None:
        for k, obs in enumerate(config.observables):
            if obs.obs_type in (ObsType.ANGULAR_SIZE_SNR,
                                ObsType.ANGULAR_SIZE_PWN,
                                ObsType.SPECTRAL_INDEX):
                continue
            if obs.obs_type != ObsType.FLUX_DENSITY:
                continue
            if obs.upper_limit:
                continue
            if obs.error is None or obs.error == 0:
                continue
            E_eV = obs._freq_hz * _h_eV
            if obs.range_lo is not None and obs.range_hi is not None:
                if predicted[k] < obs.range_lo:
                    sigma = (obs.range_lo - predicted[k]) / obs.error
                elif predicted[k] > obs.range_hi:
                    sigma = (obs.range_hi - predicted[k]) / obs.error
                else:
                    sigma = 0.0
            else:
                sigma = (obs.value - predicted[k]) / obs.error
            color = _particle_sed_color(obs.label)
            ax_res.errorbar(E_eV, sigma, yerr=1.0, fmt='o', c=color,
                            ms=4, capsize=2, zorder=5)
        ax_res.axhline(0, color='k', ls='-', lw=0.8)
        ax_res.axhline(-2, color='gray', ls='--', lw=0.5, alpha=0.5)
        ax_res.axhline(2, color='gray', ls='--', lw=0.5, alpha=0.5)
        ax_res.set_ylabel(r'$\sigma$')
        ax_res.set_xlabel('Photon Energy [eV]')
        ax_res.set_xscale('log')
        ax_res.set_xlim(photon_xlim)
        ax_res.set_ylim(-3.5, 3.5)
        ax_res.xaxis.set_major_locator(
            ticker.FixedLocator([1e-6, 1e-3, 1, 1e3, 1e6, 1e9, 1e12, 1e15]))
        ax_res.xaxis.set_major_formatter(ticker.LogFormatterSciNotation())
        ax_res.grid(alpha=0.3)

    # --- SED formatting ---
    ax_sed.set_xscale('log')
    ax_sed.set_yscale('log')
    ax_sed.set_xlim(photon_xlim)
    ax_sed.set_ylim(photon_ylim)
    ax_sed.xaxis.set_major_locator(
        ticker.FixedLocator([1e-6, 1e-3, 1, 1e3, 1e6, 1e9, 1e12, 1e15]))
    ax_sed.xaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    ax_sed.set_ylabel(r'$\nu F_\nu$ [erg cm$^{-2}$ s$^{-1}$]')
    if not show_residuals:
        ax_sed.set_xlabel('Photon Energy [eV]')

    lines1, labels1 = ax_sed.get_legend_handles_labels()
    if ax_overlay is not None:
        lines2, labels2 = ax_overlay.get_legend_handles_labels()
    else:
        lines2, labels2 = [], []
    ax_sed.legend(lines1 + lines2, labels1 + labels2, fontsize=8,
                  loc='upper left', facecolor='white', framealpha=1.0)
    ax_sed.grid(alpha=0.3)

    if ax_res is not None:
        _align_ylabels_left(fig, [ax_sed, ax_res])

    return fig, {'sed': ax_sed,
                 'particles': ax_overlay,
                 'residuals': ax_res,
                 'top_axis': ax_overlay}
