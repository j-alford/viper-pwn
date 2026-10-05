"""Broadband SED plots (fit-spec and model-result entry points)."""

from __future__ import annotations

import numpy as np

from ..observables import ObsType, flux_density_to_nufnu
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
from .data import (
    _DATASET_COLORS,
    _color_for_label,
    _draw_bowtie,
    _find_bowties,
    _plot_data_file,
    _unweighted_chi2,
)




# ===================================================================
# Main SED plot function
# ===================================================================

def plot_sed(config, theta, data_file: str = None,
             skip_datasets: set = None,
             ax=None, figsize=(10, 6),
             xlim=(1e-8, 1e15), ylim=(1e-16, 1e-10),
             title: str = None,
             show_residuals: bool = True,
             show_params: bool = False,
             # --- Customization kwargs ---
             model_color: str = 'k',
             model_lw: float = 1.5,
             model_label: str = 'Model',
             legend_fontsize: float = 13,
             legend_loc: str = 'upper left',
             params_fontsize: float = 7,
             params_loc: str = 'lower left',
             grid_alpha: float = 0.3,
             phot_bin_style: str = 'curve',
             fade_dex: float = _FADED_STEP_DEFAULT_DEX,
             apply_style: bool = True):
    """One-liner SED plot: model + data + bowties.

    For an overlaid evolved particle spectrum use
    :func:`plot_sed_with_particles`.

    Parameters
    ----------
    config : _FitSpec
        Fit specification (from build_fit_spec or run_mcmc).
    theta : array-like
        MCMC parameter vector (log-space where applicable).
    data_file : str, optional
        Path to SED data CSV; bowtie-matched datasets are auto-skipped.
    skip_datasets : set of str, optional
        Additional datasets to skip from the data file (e.g. {'VERITAS'}).
    ax : matplotlib axes, optional
        Existing axes; ignored when ``show_residuals=True``.
    figsize : tuple
        Figure size if creating new figure.
    xlim, ylim : tuple
        Axis limits.
    title : str, optional
        Plot title.
    show_residuals : bool
        If True, add a residual panel below the SED.
    show_params : bool
        If True, add a text box with decoded parameters and chi2.
    model_color : str
        Color of the model SED curve (default ``'k'``).
    model_lw : float
        Line width of the model SED curve (default 1.5).
    model_label : str
        Legend label for the model curve (default ``'Model'``).
    legend_fontsize : float
        Font size for the legend (default 13).
    legend_loc : str
        Legend location (default ``'upper left'``).
    params_fontsize : float
        Font size for the parameter text box (default 7).
    params_loc : str
        Parameter text box corner: ``'lower left'``, ``'lower right'``,
        ``'upper left'``, or ``'upper right'`` (default ``'lower left'``).
    grid_alpha : float
        Grid transparency (default 0.3).
    phot_bin_style : {'curve', 'step', 'shaded_step', 'faded_step'}, default 'curve'
        Visual representation of the photon SED.
    fade_dex : float, default 1.0
        Number of decades over which the 'faded_step' fill reaches zero.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes (or tuple of Axes if residuals)
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    if apply_style:
        _apply_sed_style()
    from ..fitting import predict_observables, log_likelihood

    _validate_bin_style(phot_bin_style)

    predicted, result = predict_observables(theta, config)

    freq = result.photon_freqs
    spec = result.photon_spectrum
    mask = spec > 0

    distance = config.all_values.get('distance', 1000)
    for i, fp in enumerate(config.free_params):
        if fp.name == 'distance':
            distance = theta[i] if not fp.log else 10**theta[i]
            break

    d_cm = distance * C.pc
    energy_eV = freq * _h_eV
    nufnu = 1e40 * (spec * freq) / (4 * np.pi * d_cm**2)

    if show_residuals:
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        gs = GridSpec(2, 1, height_ratios=[3, 1], figure=fig)
        ax = fig.add_subplot(gs[0])
        ax_res = fig.add_subplot(gs[1], sharex=ax)
        plt.setp(ax.get_xticklabels(), visible=False)
    elif ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        fig = ax.get_figure()

    # --- Model curve ---
    phot_bin_min_eV = result.phot_grid.bin_min * _h_eV
    phot_bin_max_eV = result.phot_grid.bin_max * _h_eV
    _plot_binned(
        ax, phot_bin_style,
        energy_eV[mask], nufnu[mask],
        phot_bin_min_eV[mask], phot_bin_max_eV[mask],
        fade_dex=fade_dex,
        color=model_color, ls='-', lw=model_lw,
        label=model_label, zorder=5,
    )

    # --- Detect and draw bowties ---
    bowties = _find_bowties(config.observables)
    bowtie_obs_indices = set()
    for bt in bowties:
        _draw_bowtie(ax, bt)
        bowtie_obs_indices.add(bt['si_idx'])
        bowtie_obs_indices.add(bt['fd_idx'])

    label_once = {}
    for bt in bowties:
        label_once[bt['label']] = True

    if data_file is not None:
        # --- Data file provided: use it for all SED points ---
        all_skip = set(skip_datasets or set())
        # skip datasets covered by bowties
        for bt in bowties:
            if bt['E_lo_eV'] > 100 and bt['E_hi_eV'] < 1e5:
                all_skip.add('CXO')
        _plot_data_file(ax, data_file, skip_datasets=all_skip,
                        label_once=label_once)
    else:
        # --- No data file: plot SED observables directly ---
        for i, obs in enumerate(config.observables):
            if i in bowtie_obs_indices:
                continue
            if obs.obs_type in (ObsType.ANGULAR_SIZE_SNR,
                                ObsType.ANGULAR_SIZE_PWN,
                                ObsType.SPECTRAL_INDEX):
                continue

            if obs.obs_type == ObsType.FLUX_DENSITY:
                E_eV = obs._freq_hz * _h_eV
                color = _color_for_label(obs.label)

                label_key = color
                if label_key in label_once:
                    label = ''
                else:
                    label = next((k for k in _DATASET_COLORS if k in obs.label), obs.label.split('[')[0].strip())
                    label_once[label_key] = True

                nufnu_val = flux_density_to_nufnu(
                    obs.value, obs._freq_hz, obs._flux_unit)
                if obs.upper_limit:
                    ax.errorbar(E_eV, nufnu_val,
                                yerr=nufnu_val * 0.6,
                                uplims=True, c=color, ls='none',
                                marker='o', ms=4, label=label)
                else:
                    nufnu_err = flux_density_to_nufnu(
                        obs.error, obs._freq_hz, obs._flux_unit)
                    ax.errorbar(E_eV, nufnu_val, yerr=nufnu_err,
                                c=color, ls='none', marker='o',
                                ms=4, capsize=2, label=label)

            elif obs.obs_type == ObsType.BAND_FLUX:
                # band-averaged nuFnu: F / ln(E_hi / E_lo)
                E_lo = obs._freq_lo_hz * _h_eV
                E_hi = obs._freq_hi_hz * _h_eV
                E_mid = np.sqrt(E_lo * E_hi)
                per_lnE = 1.0 / np.log(E_hi / E_lo)
                color = _color_for_label(obs.label)
                if color in label_once:
                    label = ''
                else:
                    label = next((k for k in _DATASET_COLORS if k in obs.label),
                                 obs.label.split('[')[0].strip())
                    label_once[color] = True
                y = obs.value * per_lnE
                xerr = [[E_mid - E_lo], [E_hi - E_mid]]
                if obs.upper_limit or obs.lower_limit:
                    ax.errorbar(E_mid, y, xerr=xerr, yerr=y * 0.6,
                                uplims=obs.upper_limit,
                                lolims=obs.lower_limit, c=color, ls='none',
                                marker='o', ms=4, label=label)
                else:
                    ax.errorbar(E_mid, y, xerr=xerr,
                                yerr=obs.error * per_lnE, c=color,
                                ls='none', marker='o', ms=4, capsize=2,
                                label=label)

    # --- Residual panel ---
    if show_residuals:
        res_energies = []
        res_sigmas = []
        res_colors = []
        res_markers = []
        for k, obs in enumerate(config.observables):
            if obs.obs_type in (ObsType.ANGULAR_SIZE_SNR,
                                ObsType.ANGULAR_SIZE_PWN):
                continue
            if obs.obs_type == ObsType.SPECTRAL_INDEX:
                continue
            if obs.obs_type == ObsType.FLUX_DENSITY:
                E_eV = obs._freq_hz * _h_eV
                if obs.upper_limit:
                    continue
                if obs.error is None or obs.error == 0:
                    continue
                if obs.range_lo is not None and obs.range_hi is not None:
                    mid = 0.5 * (obs.range_lo + obs.range_hi)
                    if predicted[k] < obs.range_lo:
                        sigma = (obs.range_lo - predicted[k]) / obs.error
                    elif predicted[k] > obs.range_hi:
                        sigma = (obs.range_hi - predicted[k]) / obs.error
                    else:
                        sigma = 0.0
                else:
                    sigma = (obs.value - predicted[k]) / obs.error
                res_energies.append(E_eV)
                res_sigmas.append(sigma)
                res_colors.append(_color_for_label(obs.label))
                res_markers.append('o')

        if res_energies:
            for e, s, c, m in zip(res_energies, res_sigmas,
                                  res_colors, res_markers):
                ax_res.errorbar(e, s, yerr=1.0, fmt=m, c=c, ms=4,
                                capsize=2, zorder=5)
            ax_res.axhline(0, color='k', ls='-', lw=0.8)
            ax_res.axhline(-2, color='gray', ls='--', lw=0.5, alpha=0.5)
            ax_res.axhline(2, color='gray', ls='--', lw=0.5, alpha=0.5)
            rmax = max(3.5, 1.15 * (max(abs(s) for s in res_sigmas) + 1.0))
            ax_res.set_ylim(-rmax, rmax)
            ax_res.set_ylabel(r'Residual [$\sigma$]')
            ax_res.set_xlabel('Energy [eV]')
            ax_res.set_xscale('log')
            ax_res.grid(alpha=grid_alpha)

    # --- Parameter text box ---
    if show_params:
        lines = []
        for i, fp in enumerate(config.free_params):
            val = theta[i]
            phys = 10.0 ** val if fp.log else val
            if abs(phys) >= 100:
                lines.append(f'{fp.name} = {phys:.1f}')
            elif abs(phys) >= 1:
                lines.append(f'{fp.name} = {phys:.3f}')
            else:
                lines.append(f'{fp.name} = {phys:.4g}')

        ll = log_likelihood(np.asarray(theta, dtype=float), config)
        chi2_w = -2.0 * ll
        chi2_uw = _unweighted_chi2(predicted, config)
        _ndof = len(config.observables) - len(config.free_params)
        lines.append(rf'$\chi^2$/dof = {chi2_uw:.1f}/{_ndof}')

        text = '\n'.join(lines)

        _loc_map = {
            'lower left':  (0.02, 0.02, 'left', 'bottom'),
            'lower right': (0.98, 0.02, 'right', 'bottom'),
            'upper left':  (0.02, 0.98, 'left', 'top'),
            'upper right': (0.98, 0.98, 'right', 'top'),
        }
        x, y, ha, va = _loc_map.get(params_loc, (0.02, 0.02, 'left', 'bottom'))

        ax.text(x, y, text, transform=ax.transAxes,
                fontsize=params_fontsize, family='monospace',
                verticalalignment=va, horizontalalignment=ha,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat',
                          alpha=0.8),
                zorder=10)

    # --- Formatting ---
    import matplotlib.ticker as ticker
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.set_major_locator(ticker.FixedLocator([1e-6, 1e-3, 1, 1e3, 1e6, 1e9, 1e12, 1e15]))
    ax.xaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    ax.xaxis.set_minor_locator(ticker.LogLocator(base=10, subs=[1.0], numticks=30))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if not show_residuals:
        ax.set_xlabel('Energy [eV]')
    ax.set_ylabel(r'$\nu F_\nu$ [erg s$^{-1}$ cm$^{-2}$]')
    if title is None:
        _chi2 = _unweighted_chi2(predicted, config)
        _ndof = len(config.observables) - len(config.free_params)
        ax.set_title(rf'$\chi^2$/dof = {_chi2:.1f}/{_ndof}')
    elif title:
        ax.set_title(title)
    ax.legend(fontsize=legend_fontsize, loc=legend_loc,
              facecolor='white', framealpha=1.0)
    ax.grid(alpha=grid_alpha)

    if show_residuals:
        _align_ylabels_left(fig, [ax, ax_res])
        return fig, (ax, ax_res)
    return fig, ax



def plot_model_sed(result, distance, ax=None, figsize=(10, 6),
                   model_label='SED',
                   photon_xlim=(1e-6, 1e15),
                   photon_ylim=(1e-16, 1e-10),
                   particle_ylim=(1e42, 1e48),
                   particle_xlim=(1e8, 1e15),
                   phot_bin_style='curve',
                   elec_bin_style='faded_step',
                   fade_dex=_FADED_STEP_DEFAULT_DEX,
                   apply_style=True):
    """SED + particle spectrum overlay for a model without observational data.

    For data overlays and residuals use :func:`plot_sed_with_particles`.

    Parameters
    ----------
    result : PWNModelResult
        Output of ``PWNModel.run()`` with photon and electron spectra.
    distance : float or astropy Quantity
        Source distance.  Bare floats are interpreted as parsecs.
    ax : matplotlib Axes, optional
    figsize : tuple
    model_label : str
    photon_xlim, photon_ylim, particle_ylim : tuple
        Axis limits for the photon (bottom/left) and particle (top/right)
        panels.
    particle_xlim : tuple, optional
        Top-axis limits; auto-centered when None (1 top decade = 3 bottom).
    phot_bin_style : {'curve', 'step', 'shaded_step', 'faded_step'}, default 'curve'
        Visual representation of the photon SED.  See :func:`plot_sed`.
    elec_bin_style : {'curve', 'step', 'shaded_step', 'faded_step'}, default 'faded_step'
        Visual representation of the particle spectrum.
    fade_dex : float, default 1.0
        Number of decades over which the 'faded_step' fill reaches zero.

    Returns
    -------
    fig : matplotlib Figure
    axes : dict
        Keys ``'sed'`` and ``'particles'``.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    _validate_bin_style(phot_bin_style)
    _validate_bin_style(elec_bin_style)

    if apply_style:
        _apply_sed_style()

    if hasattr(distance, 'to'):
        import astropy.units as _u
        distance_pc = float(distance.to(_u.pc).value)
    else:
        distance_pc = float(distance)
    d_cm = distance_pc * C.pc

    freq      = result.photon_freqs
    phot_spec = result.photon_spectrum
    phot_E_eV = freq * _h_eV
    nufnu     = 1e40 * (phot_spec * freq) / (4.0 * np.pi * d_cm ** 2)
    pmask     = phot_spec > 0

    grid      = result.elec_grid
    interior  = slice(1, grid.n_bins - 1)
    e_gev     = grid.bin_avg[interior]
    e_ev      = e_gev * 1e9
    e_erg     = e_ev * _eV_to_erg
    _pw, _pbmax = _particle_bin_edges(grid, interior,
                                      getattr(result, 'e_max_inj', None))
    widths_ev = _pw * 1e9
    elec_spec = result.electron_spectrum[interior]
    emask     = elec_spec > 0
    dnde      = elec_spec[emask] * 1e40 / widths_ev[emask]
    e2dnde    = e_erg[emask] ** 2 * dnde * (1.0 / _eV_to_erg)

    if ax is None:
        fig, ax_sed = plt.subplots(figsize=figsize)
        fig.subplots_adjust(left=0.10, right=0.90, top=0.88, bottom=0.12)
    else:
        ax_sed = ax
        fig = ax_sed.get_figure()

    ax_sed.set_xscale('log')
    ax_sed.set_yscale('log')
    phot_bin_min_eV = result.phot_grid.bin_min * _h_eV
    phot_bin_max_eV = result.phot_grid.bin_max * _h_eV
    _plot_binned(
        ax_sed, phot_bin_style,
        phot_E_eV[pmask], nufnu[pmask],
        phot_bin_min_eV[pmask], phot_bin_max_eV[pmask],
        fade_dex=fade_dex,
        color='k', ls='-', lw=2.25,
        label=model_label, zorder=5,
    )
    ax_sed.set_xlim(photon_xlim)
    ax_sed.set_ylim(photon_ylim)
    ax_sed.set_xlabel('Photon Energy [eV]')
    ax_sed.set_ylabel(r'$\nu F_\nu$ [erg cm$^{-2}$ s$^{-1}$]')
    ax_sed.grid(which='major', alpha=0.3)

    bottom_decades = np.log10(photon_xlim[1]) - np.log10(photon_xlim[0])
    top_decades    = bottom_decades / 3.0
    x_all = e_ev[emask]
    y_all = e2dnde
    above_floor = y_all >= particle_ylim[0]
    if above_floor.any():
        log_lo_data = np.log10(x_all[above_floor].min())
        log_hi_data = np.log10(x_all[above_floor].max())
    else:
        log_lo_data = np.log10(x_all.min())
        log_hi_data = np.log10(x_all.max())
    log_center = 0.5 * (log_lo_data + log_hi_data)

    if particle_xlim is None:
        log_top_lo = int(round(log_center - top_decades / 2))
        log_top_hi = log_top_lo + top_decades
        particle_xlim_used = (10.0 ** log_top_lo, 10.0 ** log_top_hi)
    else:
        particle_xlim_used = particle_xlim
        log_top_lo = np.log10(particle_xlim_used[0])
        log_top_hi = np.log10(particle_xlim_used[1])

    ax_overlay = fig.add_axes(ax_sed.get_position(), frameon=False)
    ax_overlay.set_xscale('log')
    ax_overlay.set_yscale('log')
    ax_overlay.set_xlim(particle_xlim_used)
    ax_overlay.set_ylim(particle_ylim)
    elec_bin_min_eV = grid.bin_min[interior] * 1e9
    elec_bin_max_eV = _pbmax * 1e9
    _plot_binned(
        ax_overlay, elec_bin_style,
        x_all, y_all,
        elec_bin_min_eV[emask], elec_bin_max_eV[emask],
        fade_dex=fade_dex,
        color='xkcd:blue', ls='-', lw=2.1,
        label=r'Particle spectrum ($e^{\pm}$)', zorder=3,
    )

    ax_overlay.xaxis.set_label_position('top')
    ax_overlay.xaxis.tick_top()
    ax_overlay.set_xlabel('Particle Energy [eV]', labelpad=10)
    tick_decades = np.arange(int(np.ceil(log_top_lo)),
                             int(np.floor(log_top_hi)) + 1)
    ax_overlay.xaxis.set_major_locator(
        ticker.FixedLocator([10.0 ** k for k in tick_decades]))
    ax_overlay.xaxis.set_major_formatter(
        ticker.LogFormatterSciNotation())
    ax_overlay.xaxis.set_minor_locator(ticker.NullLocator())

    ax_overlay.yaxis.set_label_position('right')
    ax_overlay.yaxis.tick_right()
    ax_overlay.set_ylabel(r'$E^2\,dN/dE$ [erg]', labelpad=8)

    h1, l1 = ax_sed.get_legend_handles_labels()
    h2, l2 = ax_overlay.get_legend_handles_labels()
    ax_sed.legend(h1 + h2, l1 + l2, loc='upper left', frameon=False)

    return fig, {'sed': ax_sed, 'particles': ax_overlay}
