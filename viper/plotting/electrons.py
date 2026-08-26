"""Electron energy-spectrum plots and injection overlays."""

from __future__ import annotations

import numpy as np

from .style import _ESCAPED_COLOR, _apply_sed_style




# ===================================================================
# Electron spectrum plot
# ===================================================================

def _injection_dnde(energies, params):
    """Compute the analytical injection dN/dE shape from model parameters.

    Returns unnormalized dN/dE at each energy in *energies* [GeV],
    extrapolated across the full range.
    """
    e_break = params.e_break
    p1 = params.p1
    p2 = params.p2

    dnde = np.zeros_like(energies)
    mask = energies > 0

    if e_break > 0:
        lo = mask & (energies < e_break)
        hi = mask & (energies >= e_break)
        dnde[lo] = (energies[lo] / e_break) ** (-p1)
        dnde[hi] = (energies[hi] / e_break) ** (-p2)
    else:
        dnde[mask] = energies[mask] ** (-p1)

    return dnde


def _fit_electron_powerlaws(energies, dnde, n_smooth=5, e_break_inj=None):
    """Find the synchrotron cooling break and fit power laws on each side.

    Works in log-log space; the break is where the local spectral
    index changes most rapidly.

    Parameters
    ----------
    energies : ndarray
        Bin-center energies [MeV].
    dnde : ndarray
        dN/dE values [MeV^{-1}].
    n_smooth : int
        Half-width of the boxcar smoothing window for the local index.
    e_break_inj : float, optional
        Injection break energy [MeV]; caps the upper power-law fit.

    Returns
    -------
    dict with keys:
        E_break : float  -- cooling break energy [MeV]
        p_lo    : float  -- spectral index below the break
        p_hi    : float  -- spectral index above the break
        fit_lo  : (e_array, dnde_array) -- power law line below break
        fit_hi  : (e_array, dnde_array) -- power law line above break
    Returns None if the spectrum is too sparse to fit.
    """
    mask = dnde > 0
    if np.sum(mask) < 20:
        return None

    log_e = np.log10(energies[mask])
    log_n = np.log10(dnde[mask])

    local_idx = -np.gradient(log_n, log_e)

    kernel = np.ones(2 * n_smooth + 1) / (2 * n_smooth + 1)
    if len(local_idx) > len(kernel):
        smooth_idx = np.convolve(local_idx, kernel, mode='same')
    else:
        smooth_idx = local_idx

    # middle 80% avoids edge effects
    n = len(smooth_idx)
    i_lo = n // 10
    i_hi = n - n // 10
    d_idx = np.gradient(smooth_idx)
    search = d_idx[i_lo:i_hi]
    if len(search) == 0:
        return None
    break_i = i_lo + np.argmax(search)
    e_break = 10.0 ** log_e[break_i]

    # fit away from the break
    margin = max(3, n // 20)
    lo_slice = slice(i_lo, max(i_lo + 1, break_i - margin))

    # cap upper fit at injection break
    i_hi_fit = i_hi
    if e_break_inj is not None and e_break_inj > 0:
        log_e_brk_inj = np.log10(e_break_inj)
        above_inj = np.where(log_e > log_e_brk_inj)[0]
        if len(above_inj) > 0 and above_inj[0] < i_hi:
            i_hi_fit = above_inj[0]
    hi_slice = slice(min(break_i + margin, i_hi_fit - 1), i_hi_fit)

    if log_e[lo_slice].size < 3 or log_e[hi_slice].size < 3:
        return None

    c_lo = np.polyfit(log_e[lo_slice], log_n[lo_slice], 1)
    c_hi = np.polyfit(log_e[hi_slice], log_n[hi_slice], 1)

    p_lo = -c_lo[0]   # spectral index = -slope
    p_hi = -c_hi[0]

    e_fit_lo = np.geomspace(10.0 ** log_e[lo_slice][0],
                            10.0 ** log_e[lo_slice][-1], 50)
    e_fit_hi = np.geomspace(10.0 ** log_e[hi_slice][0],
                            10.0 ** log_e[hi_slice][-1], 50)
    dnde_fit_lo = 10.0 ** np.polyval(c_lo, np.log10(e_fit_lo))
    dnde_fit_hi = 10.0 ** np.polyval(c_hi, np.log10(e_fit_hi))

    # third segment above injection break
    p_top = None
    fit_top = None
    if e_break_inj is not None and e_break_inj > 0:
        log_e_brk_inj = np.log10(e_break_inj)
        above_inj = np.where((log_e > log_e_brk_inj) & (log_e < log_e[-1]))[0]
        if len(above_inj) >= 5:
            margin_top = max(2, len(above_inj) // 10)
            top_slice = slice(above_inj[margin_top], above_inj[-margin_top])
            if log_e[top_slice].size >= 3:
                c_top = np.polyfit(log_e[top_slice], log_n[top_slice], 1)
                p_top = -c_top[0]
                e_fit_top = np.geomspace(10.0 ** log_e[top_slice][0],
                                         10.0 ** log_e[top_slice][-1], 50)
                dnde_fit_top = 10.0 ** np.polyval(c_top, np.log10(e_fit_top))
                fit_top = (e_fit_top, dnde_fit_top)

    return {
        'E_break': e_break,
        'p_lo': p_lo,
        'p_hi': p_hi,
        'fit_lo': (e_fit_lo, dnde_fit_lo),
        'fit_hi': (e_fit_hi, dnde_fit_hi),
        'p_top': p_top,
        'fit_top': fit_top,
    }


def plot_electron_spectrum(
    result,
    times=None,
    energy_weighted: bool = True,
    params=None,
    fit_powerlaws: bool = False,
    label_indices: bool = False,
    show_escaped: bool = False,
    ax=None,
    figsize=(10, 6),
    xlim=None,
    ylim=None,
    title: str = None,
    cmap: str = 'viridis',
    alpha: float = 0.7,
    apply_style: bool = True,
):
    """Plot the electron energy spectrum from a model run.

    Parameters
    ----------
    result : PWNModelResult
        Output from evolve_pwn() or PWNModel.run().
    times : None, list of float, or 'all'
        None: final spectrum only.  List of ages [yr]: nearest
        snapshot each (needs elecinfo, elec_step > 0).  'all':
        every snapshot.
    energy_weighted : bool
        If True, plot E^2 dN/dE [GeV] instead of dN/dE [GeV^{-1}].
    params : PWNScalarParams, optional
        Overlay the analytical injection shape, normalized at low energies.
    fit_powerlaws : bool
        If True, fit power laws around the cooling break and annotate
        the spectral indices.
    label_indices : bool
        If True and params is provided, annotate p1 and p2.
    show_escaped : bool
        If True, overlay the cumulative escaped ("halo") spectrum.
        Raises ValueError if escape was disabled.
    ax : matplotlib Axes, optional
        Plot onto existing axes.
    figsize : tuple
        Figure size if creating a new figure.
    xlim, ylim : tuple, optional
        Axis limits.  Sensible defaults are chosen automatically.
    title : str, optional
        Plot title.
    cmap : str
        Colormap name for multi-epoch curves (default: 'viridis').
    alpha : float
        Line transparency for snapshot curves.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib as mpl
    if apply_style:
        _apply_sed_style()

    if result.elec_grid is None:
        raise ValueError("No electron grid in result -- did the model run?")

    grid = result.elec_grid
    # skip underflow/overflow bins
    interior = slice(1, grid.n_bins - 1)
    energies = grid.bin_avg[interior] * 1.0e9      # GeV -> eV
    widths = grid.bin_width[interior] * 1.0e9       # GeV -> eV

    # -- Collect (time_label, spectrum) pairs to plot --
    curves = []

    if times is None:
        curves.append((None, result.electron_spectrum[interior]))

    elif times == 'all':
        if not result.elecinfo:
            raise ValueError(
                "No elecinfo snapshots. Re-run with elec_step > 0."
            )
        for t, spec in result.elecinfo:
            curves.append((t, spec[interior]))

    else:
        if not result.elecinfo:
            raise ValueError(
                "No elecinfo snapshots. Re-run with elec_step > 0."
            )
        snap_times = np.array([t for t, _ in result.elecinfo])
        for t_req in times:
            idx = np.argmin(np.abs(snap_times - t_req))
            t_snap, spec = result.elecinfo[idx]
            curves.append((t_snap, spec[interior]))

    # -- Create figure --
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        fig = ax.get_figure()

    # -- Compute final dN/dE for overlays (injection + powerlaw fits) --
    final_spec = result.electron_spectrum[interior]
    final_mask = final_spec > 0
    final_dnde = np.zeros_like(final_spec)
    final_dnde[final_mask] = final_spec[final_mask] / widths[final_mask]

    # -- Compute y-values and plot --
    if len(curves) == 1:
        t_label, spec = curves[0]
        mask = spec > 0
        dnde = spec[mask] / widths[mask]
        e = energies[mask]
        if energy_weighted:
            y = e**2 * dnde
        else:
            y = dnde
        label = f't = {t_label:.0f} yr' if t_label is not None else 'Final'
        ax.plot(e, y, 'k-', lw=1.5, label=label)

    else:
        t_vals = [t for t, _ in curves]
        t_min, t_max = min(t_vals), max(t_vals)
        norm = mcolors.LogNorm(vmin=max(t_min, 1.0), vmax=t_max)
        cm = plt.get_cmap(cmap)

        for t_label, spec in curves:
            mask = spec > 0
            if not np.any(mask):
                continue
            dnde = spec[mask] / widths[mask]
            e = energies[mask]
            if energy_weighted:
                y = e**2 * dnde
            else:
                y = dnde
            color = cm(norm(max(t_label, 1.0)))
            ax.plot(e, y, color=color, lw=1.0, alpha=alpha)

        sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label('Age [yr]')

    # -- Escaped (halo) population overlay (cumulative reservoir) --
    if show_escaped:
        esc_full = result.escaped_spectrum
        if esc_full is None or not np.any(np.asarray(esc_full) > 0):
            raise ValueError(
                "show_escaped=True but result.escaped_spectrum is empty "
                "(escape_model='off' or escape disabled). Re-run with "
                "escape_model != 'off'."
            )
        esc = np.asarray(esc_full)[interior]
        esc_mask = esc > 0
        if np.any(esc_mask):
            esc_dnde = esc[esc_mask] / widths[esc_mask]
            e_esc = energies[esc_mask]
            y_esc = e_esc ** 2 * esc_dnde if energy_weighted else esc_dnde
            ax.plot(e_esc, y_esc, color=_ESCAPED_COLOR, ls='--', lw=1.6,
                    label='Escaped (cumulative)')

    # -- Injection spectrum overlay --
    if params is not None:
        energies_gev = energies / 1.0e9  # back to GeV for params
        inj_dnde = _injection_dnde(energies_gev, params)
        inj_mask = inj_dnde > 0
        if np.any(inj_mask & final_mask):
            # normalize near e_min (injection-dominated)
            e_min_ev = params.e_min * 1.0e9
            near_emin = (inj_mask & final_mask
                         & (energies > e_min_ev * 0.8)
                         & (energies < e_min_ev * 3.0))
            if np.sum(near_emin) < 3:
                both = inj_mask & final_mask & (energies >= e_min_ev)
                ref_idx = np.where(both)[0][:10]
            else:
                ref_idx = np.where(near_emin)[0]
            if len(ref_idx) > 0:
                scale = np.median(
                    final_dnde[ref_idx] / inj_dnde[ref_idx]
                )
                inj_scaled = inj_dnde * scale
                e_inj = energies[inj_mask]
                y_inj = inj_scaled[inj_mask]
                if energy_weighted:
                    y_inj = e_inj**2 * y_inj
                ax.plot(e_inj, y_inj, 'b--', lw=1.2, alpha=0.6,
                        label='Injected')

                if label_indices:
                    e_break_ev = params.e_break * 1.0e9
                    lo_pts = inj_mask & (energies < e_break_ev) & (energies > e_min_ev * 0.5)
                    if np.sum(lo_pts) > 2:
                        idx_lo = np.where(lo_pts)[0][len(np.where(lo_pts)[0]) // 2]
                        y_label = inj_scaled[idx_lo]
                        if energy_weighted:
                            y_label = energies[idx_lo]**2 * y_label
                        ax.annotate(f'p1 = {params.p1:.2f}',
                                    xy=(energies[idx_lo], y_label),
                                    xytext=(10, 10), textcoords='offset points',
                                    fontsize=9, color='blue', alpha=0.8)
                    hi_pts = inj_mask & (energies > e_break_ev)
                    if np.sum(hi_pts) > 2:
                        idx_hi = np.where(hi_pts)[0][len(np.where(hi_pts)[0]) // 2]
                        y_label = inj_scaled[idx_hi]
                        if energy_weighted:
                            y_label = energies[idx_hi]**2 * y_label
                        ax.annotate(f'p2 = {params.p2:.2f}',
                                    xy=(energies[idx_hi], y_label),
                                    xytext=(10, -15), textcoords='offset points',
                                    fontsize=9, color='blue', alpha=0.8)

    # -- Power law fits around cooling break --
    if fit_powerlaws:
        e_brk_inj = params.e_break * 1.0e9 if params is not None else None
        fit = _fit_electron_powerlaws(energies, final_dnde,
                                      e_break_inj=e_brk_inj)
        if fit is not None:
            e_lo, dnde_lo = fit['fit_lo']
            e_hi, dnde_hi = fit['fit_hi']
            # 3x offset keeps fit lines visible
            dnde_lo = dnde_lo * 3.0
            dnde_hi = dnde_hi * 3.0
            if energy_weighted:
                dnde_lo = e_lo**2 * dnde_lo
                dnde_hi = e_hi**2 * dnde_hi
            ax.plot(e_lo, dnde_lo, color='0.3', ls='--', lw=1.8,
                    alpha=0.8, label=f'p = {fit["p_lo"]:.2f}')
            ax.plot(e_hi, dnde_hi, color='0.3', ls='--', lw=1.8,
                    alpha=0.8, label=f'p = {fit["p_hi"]:.2f}')
            e_brk = fit['E_break']  # eV
            if e_brk >= 1e12:
                brk_str = f'{e_brk/1e12:.1f} TeV'
            elif e_brk >= 1e9:
                brk_str = f'{e_brk/1e9:.1f} GeV'
            elif e_brk >= 1e6:
                brk_str = f'{e_brk/1e6:.0f} MeV'
            else:
                brk_str = f'{e_brk:.0f} eV'
            ax.axvline(e_brk, color='0.5', ls=':',
                       alpha=0.5, label=f'Break: {brk_str}')
            if fit['fit_top'] is not None:
                e_top, dnde_top = fit['fit_top']
                dnde_top = dnde_top * 3.0
                if energy_weighted:
                    dnde_top = e_top**2 * dnde_top
                ax.plot(e_top, dnde_top, color='0.3', ls='--', lw=1.8,
                        alpha=0.8, label=f'p = {fit["p_top"]:.2f}')

    # -- Formatting --
    import matplotlib.ticker as ticker

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.xaxis.set_major_locator(ticker.FixedLocator([1e6, 1e9, 1e12, 1e15]))
    ax.xaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    ax.xaxis.set_minor_locator(ticker.LogLocator(base=10, subs=[1.0],
                                                  numticks=20))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())

    if energy_weighted:
        ax.set_ylabel(r'$E^2 \, dN/dE$ [eV]')
    else:
        ax.set_ylabel(r'$dN/dE$ [eV$^{-1}$]')

    ax.set_xlabel('Energy [eV]')

    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        all_x = np.concatenate([line.get_xdata() for line in ax.get_lines()
                                if len(line.get_xdata()) > 0])
        all_x = all_x[all_x > 0]
        if len(all_x) > 0:
            ax.set_xlim(all_x.min() * 0.5, all_x.max() * 2)
    if ylim is not None:
        ax.set_ylim(ylim)
    if title:
        ax.set_title(title)

    ax.legend(fontsize=8)
    ax.grid(which='major', alpha=0.3)

    return fig, ax
