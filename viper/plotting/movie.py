"""Time-evolution SED movie rendering."""

from __future__ import annotations

import numpy as np

from ..observables import ObsType
from .. import constants as C
from .style import (
    _particle_bin_edges,
    _ESCAPED_COLOR,
    _FADED_STEP_DEFAULT_DEX,
    _SHADED_STEP_FILL_ALPHA,
    _align_ylabels_left,
    _apply_sed_style,
    _eV_to_erg,
    _faded_step_offsets,
    _h_eV,
    _shaded_step_verts,
    _step_slab_verts,
    _validate_bin_style,
)
from .data import _DATASET_COLORS, _color_for_label

# units: yr·TeV and yr·TeV·µG²
_COOL_C_IC   = 1.196e6
_COOL_C_SYNC = 1.253e7


def _cooling_timescale_yr(E_TeV, B_uG):
    """Combined IC (CMB-only) + synchrotron cooling timescale in years."""
    inv_tau_ic_yr   = E_TeV / _COOL_C_IC
    inv_tau_sync_yr = E_TeV * (B_uG ** 2) / _COOL_C_SYNC
    return 1.0 / (inv_tau_ic_yr + inv_tau_sync_yr)




def _interp_spec(t, t_arr, spec_list):
    """Linearly interpolate a list of per-bin spectra at time ``t``.

    Endpoints are clamped: ``t`` outside ``[t_arr[0], t_arr[-1]]``
    returns the nearest snapshot.
    """
    if t <= t_arr[0]:
        return spec_list[0].copy()
    if t >= t_arr[-1]:
        return spec_list[-1].copy()
    i_hi = int(np.searchsorted(t_arr, t, side='left'))
    i_hi = min(max(i_hi, 1), len(t_arr) - 1)
    i_lo = i_hi - 1
    alpha = (t - t_arr[i_lo]) / (t_arr[i_hi] - t_arr[i_lo])
    return (1.0 - alpha) * spec_list[i_lo] + alpha * spec_list[i_hi]


def _interp_dyn(t, t_arr, dyn_list, field_names, DynState):
    """Linearly interpolate a list of ``DynState`` at time ``t``."""
    if t <= t_arr[0]:
        return dyn_list[0]
    if t >= t_arr[-1]:
        return dyn_list[-1]
    i_hi = int(np.searchsorted(t_arr, t, side='left'))
    i_hi = min(max(i_hi, 1), len(t_arr) - 1)
    i_lo = i_hi - 1
    alpha = (t - t_arr[i_lo]) / (t_arr[i_hi] - t_arr[i_lo])
    out = DynState()
    for name in field_names:
        v_lo = getattr(dyn_list[i_lo], name)
        v_hi = getattr(dyn_list[i_hi], name)
        setattr(out, name, (1.0 - alpha) * v_lo + alpha * v_hi)
    return out


def _uniform_time_snapshots(result, dt_yr, *, need_dyn, need_esc=False):
    """Resample (rad/elec/dyn/esc)info onto a uniform time grid.

    Returns four new lists (rad, elec, dyn, esc) sampled at uniform
    ``dt_yr`` intervals; ``result`` is not mutated.  ``dt_yr`` may be
    ``'auto'`` (``(t_max - t_min) / 200``) or a positive float in
    years.  When ``need_dyn`` (``need_esc``) is False the returned
    list is empty.
    """
    from dataclasses import fields as _dc_fields
    from ..evolution import DynState

    t_rad  = np.array([t for t, _ in result.radinfo], dtype=float)
    t_elec = np.array([t for t, _ in result.elecinfo], dtype=float)
    t_min  = float(t_rad[0])
    t_max  = float(t_rad[-1])

    if isinstance(dt_yr, str):
        if dt_yr != 'auto':
            raise ValueError(
                f"uniform_dt must be 'auto' or a positive float, "
                f"got {dt_yr!r}")
        dt = (t_max - t_min) / 200.0
    else:
        dt = float(dt_yr)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"uniform_dt must be > 0, got {dt_yr!r}")
    if t_max <= t_min:
        raise ValueError(
            "snapshot times do not span a positive interval; "
            "cannot resample to uniform grid")

    n_steps = int(np.floor((t_max - t_min) / dt))
    target_times = t_min + np.arange(n_steps + 1) * dt
    if target_times[-1] < t_max:
        target_times = np.append(target_times, t_max)

    spec_rad  = [s for _, s in result.radinfo]
    spec_elec = [s for _, s in result.elecinfo]
    if need_dyn:
        t_dyn = np.array([d.time for d in result.dyninfo], dtype=float)
        dyn_field_names = [f.name for f in _dc_fields(DynState)]
    if need_esc:
        t_esc    = np.array([t for t, _ in result.escinfo], dtype=float)
        spec_esc = [s for _, s in result.escinfo]

    radinfo_new  = []
    elecinfo_new = []
    dyninfo_new  = []
    escinfo_new  = []
    for t in target_times:
        radinfo_new.append(
            (float(t), _interp_spec(t, t_rad, spec_rad)))
        elecinfo_new.append(
            (float(t), _interp_spec(t, t_elec, spec_elec)))
        if need_dyn:
            dyninfo_new.append(
                _interp_dyn(t, t_dyn, result.dyninfo,
                            dyn_field_names, DynState))
        if need_esc:
            escinfo_new.append(
                (float(t), _interp_spec(t, t_esc, spec_esc)))
    return radinfo_new, elecinfo_new, dyninfo_new, escinfo_new


def _format_time_yr(t_yr):
    """Human-readable time stamp for evolution movies."""
    if t_yr < 1000:
        return f't = {t_yr:.0f} yr'
    if t_yr < 1e6:
        return f't = {t_yr / 1e3:.2f} kyr'
    return f't = {t_yr / 1e6:.2f} Myr'


def plot_model_sed_movie(result, distance, outfile='evolution.mp4',
                         fps=15, model_label='SED',
                         photon_xlim=(1e-6, 1e15),
                         photon_ylim=(1e-15, 1e-9),
                         particle_xlim=(1e8, 1e15),
                         particle_ylim=(1e42, 1e48),
                         figsize=None,
                         start_frame=0, end_frame=None,
                         dpi=100, bitrate=-1, verbose=True,
                         side_panel=False, model=None, show_injection=False,
                         uniform_dt=None,
                         observables=None, show_data=False,
                         residual_panel=False,
                         phot_bin_style='curve',
                         elec_bin_style='faded_step',
                         inj_bin_style='faded_step',
                         esc_bin_style='faded_step',
                         fade_dex=_FADED_STEP_DEFAULT_DEX,
                         debug_cooling=False,
                         show_escaped=False,
                         reference_radii=None,
                         apply_style=True):
    """Render an MP4/GIF of the SED + particle spectrum evolving.

    Per-frame artists: photon SED, particle spectrum, optional side
    panel with PWN/SNR/RS disks and a radius/B/velocity readout.
    Requires snapshots: call ``model.run(dyn_step=N, elec_step=N,
    phot_step=N)`` with all three set equal for the side panel.

    Parameters
    ----------
    result : PWNModelResult
    distance : float or astropy Quantity
    outfile : str
        .mp4 -> ffmpeg, .gif -> pillow.
    fps, model_label, photon_xlim, photon_ylim : standard
    particle_xlim : tuple
        Fixed top-axis range; default spans 7 decades to keep the
        1:3 photon:particle decade alignment.
    particle_ylim : tuple
    figsize : tuple, optional.  Defaults: (10, 6) plain;
        (14, 6) with side_panel; (10, 8) with residual_panel;
        (14, 8) with both.
    start_frame, end_frame, dpi, bitrate, verbose : standard
    side_panel : bool
    reference_radii : list of dict, optional
        Static rings drawn on the side panel for measured sizes, e.g.
        ``[{'radius': 1.19, 'label': 'PWN (obs.)'},
        {'r_lo': 6.5, 'r_hi': 7.7, 'label': 'SNR (obs.)'}]`` with radii
        in pc.  A ``radius`` entry is a thin ring; an ``r_lo``/``r_hi``
        entry is a thin annulus (a measured range).  Optional keys
        ``color`` (default black), ``ls`` (default ``'--'``), ``lw``
        (default 1.0), and ``shade_lo``/``shade_hi`` (pc) to fill a
        light annulus for the uncertainty (``shade_alpha``, default
        0.18).  Ignored unless ``side_panel=True``.
    model : PWNModel, optional
        Required when ``show_injection=True``; supplies the spindown
        and injection-spectrum parameters.
    show_injection : bool
        If True, overlay a cumulative-injection curve on the particle
        panel.  Requires ``vel_psr == 0``; raises NotImplementedError
        for moving pulsars.
    uniform_dt : None, 'auto', or float
        None (default): animate the recorded snapshots directly.
        Float [yr] or ``'auto'`` (``(t_max - t_min) / 200``):
        resample onto a uniform time grid by linear interpolation;
        ``result`` is not modified.
    observables : list of Observable, optional
        Required when ``show_data=True`` or ``residual_panel=True``.
        Only ``FLUX_DENSITY`` entries are used; other types are
        silently skipped.
    show_data : bool
        If True, overlay flux-density observables as static points.
        Implied when ``residual_panel=True``.
    residual_panel : bool
        If True, add a lower panel of per-frame ``(data - model) /
        sigma`` residuals; upper limits contribute none.  Y-range
        fixed to ±5σ.
    phot_bin_style : {'curve', 'step', 'shaded_step', 'faded_step'}, default 'curve'
        Visual representation of the photon SED.  See :func:`plot_sed`.
    elec_bin_style : {'curve', 'step', 'shaded_step', 'faded_step'}, default 'faded_step'
        Visual representation of the total particle spectrum.
    inj_bin_style : {'curve', 'step', 'shaded_step', 'faded_step'}, default 'faded_step'
        Style of the cumulative-injection overlay (``show_injection=True``).
    esc_bin_style : {'curve', 'step', 'shaded_step', 'faded_step'}, default 'faded_step'
        Style of the escaped-particle overlay (``show_escaped=True``).
    fade_dex : float, default 1.0
        Number of decades over which the 'faded_step' fill reaches zero.
    debug_cooling : bool, default False
        If True, mark in red the particle bins whose IC + synchrotron
        cooling time is shorter than the snapshot-to-snapshot ``dt``.
    show_escaped : bool, default False
        If True, overlay the cumulative escaped ("halo") spectrum on
        the particle panel.  Requires ``result.escinfo``
        (``escape_model != 'off'``); raises ValueError otherwise.

    Returns
    -------
    str
    """
    import os
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.animation import FuncAnimation
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Circle, Wedge
    from matplotlib.lines import Line2D
    from viper.particles import inject_particles

    _validate_bin_style(phot_bin_style)
    _validate_bin_style(elec_bin_style)
    _validate_bin_style(inj_bin_style)
    _validate_bin_style(esc_bin_style)

    _faded_offsets = _faded_step_offsets(fade_dex)

    if debug_cooling and not result.dyninfo:
        raise ValueError(
            "debug_cooling=True requires result.dyninfo with bpwn values; "
            "got an empty dyninfo. Run model.run() with dyn_step>=1."
        )

    if not result.radinfo:
        raise ValueError(
            "result.radinfo is empty.  Call model.run(phot_step=N) "
            "with N>0 to record photon snapshots."
        )
    if not result.elecinfo:
        raise ValueError(
            "result.elecinfo is empty.  Call model.run(elec_step=N) "
            "with N>0 to record electron snapshots."
        )
    if len(result.radinfo) != len(result.elecinfo):
        raise ValueError(
            f"snapshot-count mismatch: {len(result.radinfo)} photon "
            f"vs {len(result.elecinfo)} electron.  Set "
            f"elec_step == phot_step."
        )
    if show_escaped:
        if not result.escinfo:
            raise ValueError(
                "show_escaped=True requires escaped-particle snapshots, "
                "but result.escinfo is empty.  Re-run model.run(elec_step=N) "
                "with escape_model != 'off'."
            )
        if len(result.escinfo) != len(result.elecinfo):
            raise ValueError(
                f"snapshot-count mismatch: {len(result.escinfo)} escaped "
                f"vs {len(result.elecinfo)} electron."
            )
    if side_panel and len(result.dyninfo) != len(result.radinfo):
        raise ValueError(
            f"side_panel=True requires dyn/elec/phot alignment: "
            f"{len(result.dyninfo)} dyn vs "
            f"{len(result.radinfo)} phot. "
            f"Set dyn_step == elec_step == phot_step."
        )
    if show_injection and model is None:
        raise ValueError(
            "show_injection=True requires the `model` kwarg "
            "(PWNModel instance used to produce `result`)."
        )
    if show_injection and float(
            model._parameters['vel_psr'].value) != 0.0:
        raise NotImplementedError(
            "show_injection=True currently requires vel_psr == 0.  "
            "With a moving pulsar, the rad_psr > rad_pwn injection "
            "cutoff would need per-snapshot reconstruction."
        )
    if (show_injection
            and float(model._parameters['braking_index'].value) == 1.0):
        raise ValueError(
            "show_injection=True requires braking_index != 1 "
            "(cumulative-injection formula has a different branch "
            "for n == 1 that is not implemented here)."
        )
    if (show_data or residual_panel) and observables is None:
        raise ValueError(
            "show_data=True or residual_panel=True requires "
            "the `observables` kwarg (list of Observable)."
        )
    if residual_panel:
        show_data = True

    if uniform_dt is None:
        radinfo  = result.radinfo
        elecinfo = result.elecinfo
        dyninfo  = result.dyninfo
        escinfo  = result.escinfo if show_escaped else []
    else:
        radinfo, elecinfo, dyninfo, escinfo = _uniform_time_snapshots(
            result, uniform_dt, need_dyn=side_panel, need_esc=show_escaped)

    if apply_style:
        _apply_sed_style()

    PARTICLE_COLOR = 'xkcd:blue'
    SNR_COLOR      = '0.65'
    PWN_COLOR      = 'xkcd:blue'

    if hasattr(distance, 'to'):
        import astropy.units as _u
        distance_pc = float(distance.to(_u.pc).value)
    else:
        distance_pc = float(distance)
    d_cm = distance_pc * C.pc

    freq      = result.phot_grid.bin_avg
    phot_E_eV = freq * _h_eV
    phot_bin_min_eV = result.phot_grid.bin_min * _h_eV
    phot_bin_max_eV = result.phot_grid.bin_max * _h_eV
    grid      = result.elec_grid
    interior  = slice(1, grid.n_bins - 1)
    e_gev     = grid.bin_avg[interior]
    e_ev      = e_gev * 1e9
    e_erg     = e_ev * _eV_to_erg
    _pw, _pbmax = _particle_bin_edges(grid, interior,
                                      getattr(result, 'e_max_inj', None))
    widths_ev = _pw * 1e9
    elec_bin_min_eV = grid.bin_min[interior] * 1e9
    elec_bin_max_eV = _pbmax * 1e9

    if debug_cooling:
        _e_tev_per_bin = e_gev / 1e3
        _dyn_times_yr = np.array([d.time for d in result.dyninfo], dtype=float)
        _dyn_bpwn_uG  = np.array([d.bpwn for d in result.dyninfo], dtype=float)
    else:
        _e_tev_per_bin = _dyn_times_yr = _dyn_bpwn_uG = None

    if show_injection:
        p = model._parameters
        _eta_g    = p['eta_g'].value
        _eta_b    = p['eta_b'].value
        _esd_40   = model.esd_40
        _n_brake  = p['braking_index'].value
        _tau_sd   = p['tau_sd'].value
        _s_idx    = (_n_brake + 1.0) / (_n_brake - 1.0)
        _emin     = p['e_min'].value
        _emax     = p['e_max'].value
        _ebrk     = p['e_break'].value
        _p1       = p['p1'].value
        _p2       = p['p2'].value
        _fmax     = p['f_max'].value
        _kTmax    = p['kT_max'].value
        _unit_grid = result.elec_grid.copy()
        _unit_grid.reset_number(0.0)
        inject_particles(_unit_grid, energy_40=1.0,
                         e_min=_emin, e_max=_emax, e_break=_ebrk,
                         p1=_p1, p2=_p2,
                         f_max=_fmax, kT_max=_kTmax,
                         injection_model=getattr(model, 'injection_model',
                                                 'broken_powerlaw'))
        _unit_inj = _unit_grid.number.copy()

        # bins straddling e_min/e_max: partial widths
        _eff_widths_ev = widths_ev.copy()
        _emin_ev = _emin * 1e9
        _emax_ev = _emax * 1e9
        _bmin_ev = grid.bin_min[interior] * 1e9
        _bmax_ev = grid.bin_max[interior] * 1e9
        _at_min = (_bmin_ev < _emin_ev) & (_bmax_ev >= _emin_ev)
        _at_max = (_bmin_ev <= _emax_ev) & (_bmax_ev > _emax_ev)
        _eff_widths_ev[_at_min] = _bmax_ev[_at_min] - _emin_ev
        _eff_widths_ev[_at_max] = _emax_ev - _bmin_ev[_at_max]

        def _e_cum_40(t_yr):
            # cumulative injected energy [1e40 erg]; vel_psr == 0 only
            base = ((1.0 - _eta_g) * _esd_40 / (1.0 - _s_idx)
                    * ((1.0 + t_yr / _tau_sd) ** (1.0 - _s_idx)
                       - 1.0))
            return base * (1.0 - _eta_b)
    else:
        _unit_inj = None
        _e_cum_40 = None
        _eff_widths_ev = None

    n_snaps = len(radinfo)
    if end_frame is None:
        end_frame = n_snaps
    frame_indices = list(range(start_frame,
                               min(end_frame, n_snaps)))
    if not frame_indices:
        raise ValueError(
            "No frames to render after applying start/end_frame."
        )

    if figsize is None:
        if side_panel and residual_panel:
            figsize = (14, 8)
        elif side_panel:
            figsize = (14, 6)
        elif residual_panel:
            figsize = (10, 8)
        else:
            figsize = (10, 6)
    fig = plt.figure(figsize=figsize)
    if residual_panel and side_panel:
        gs = fig.add_gridspec(2, 2, width_ratios=[10, 4],
                              height_ratios=[3, 1],
                              wspace=0.60, hspace=0.08,
                              left=0.10, right=0.97,
                              top=0.88, bottom=0.10)
        ax_sed  = fig.add_subplot(gs[0, 0])
        ax_res  = fig.add_subplot(gs[1, 0], sharex=ax_sed)
        ax_side = fig.add_subplot(gs[:, 1])
    elif residual_panel:
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1],
                              hspace=0.08,
                              left=0.10, right=0.92,
                              top=0.88, bottom=0.10)
        ax_sed  = fig.add_subplot(gs[0, 0])
        ax_res  = fig.add_subplot(gs[1, 0], sharex=ax_sed)
        ax_side = None
    elif side_panel:
        gs = fig.add_gridspec(1, 2, width_ratios=[10, 4],
                              wspace=0.60,
                              left=0.10, right=0.97,
                              top=0.85, bottom=0.15)
        ax_sed  = fig.add_subplot(gs[0, 0])
        ax_side = fig.add_subplot(gs[0, 1])
        ax_res  = None
    else:
        ax_sed = fig.add_subplot(111)
        fig.subplots_adjust(left=0.10, right=0.92,
                            top=0.85, bottom=0.15)
        ax_side = None
        ax_res  = None

    ax_sed.set_xscale('log')
    ax_sed.set_yscale('log')
    ax_sed.set_xlim(photon_xlim)
    ax_sed.set_ylim(photon_ylim)
    ax_sed.set_xlabel('Photon Energy [eV]')
    ax_sed.set_ylabel(r'$\nu F_\nu$ [erg cm$^{-2}$ s$^{-1}$]')
    ax_sed.grid(which='major', alpha=0.3)

    log_top_lo = np.log10(particle_xlim[0])
    log_top_hi = np.log10(particle_xlim[1])
    ax_overlay = fig.add_axes(ax_sed.get_position(), frameon=False)
    ax_overlay.set_xscale('log')
    ax_overlay.set_yscale('log')
    ax_overlay.set_xlim(particle_xlim)
    ax_overlay.set_ylim(particle_ylim)
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

    def _make_line(ax, style, *, color, lw, label, zorder):
        """Return (Line2D, list[PolyCollection], style) for a per-frame curve.

        The fill list is empty for 'curve'/'step', one polygon for
        'shaded_step', and K polygons for 'faded_step'.
        """
        drawstyle = ('steps-mid'
                     if style in ('step', 'shaded_step', 'faded_step')
                     else 'default')
        (ln,) = ax.plot(
            [], [], color=color, ls='-', lw=lw,
            drawstyle=drawstyle, label=label, zorder=zorder)
        fills = []
        if style == 'shaded_step':
            fill = PolyCollection(
                [], alpha=_SHADED_STEP_FILL_ALPHA,
                facecolor=color, edgecolor='none', zorder=zorder - 0.5)
            ax.add_collection(fill)
            fills.append(fill)
        elif style == 'faded_step':
            _, _, alphas = _faded_offsets
            for a in alphas:
                if a <= 0:
                    continue
                fill = PolyCollection(
                    [], alpha=float(a), facecolor=color,
                    edgecolor='none', zorder=zorder - 0.5)
                ax.add_collection(fill)
                fills.append(fill)
        return (ln, fills, style)

    line_sed = _make_line(
        ax_sed, phot_bin_style,
        color='k', lw=2.25, label=model_label, zorder=5)
    if show_injection:
        line_inj = _make_line(
            ax_overlay, inj_bin_style,
            color='xkcd:grey', lw=1.2,
            label='Cumulative injection', zorder=2)
    else:
        line_inj = None
    line_elec = _make_line(
        ax_overlay, elec_bin_style,
        color=PARTICLE_COLOR, lw=2.1,
        label=r'Particle spectrum ($e^{\pm}$)', zorder=3)
    if show_escaped:
        line_esc = _make_line(
            ax_overlay, esc_bin_style,
            color=_ESCAPED_COLOR, lw=1.7,
            label=r'Escaped $e^{\pm}$ (cumulative)', zorder=2.5)
    else:
        line_esc = None
    if debug_cooling:
        # NaNs split the red line at non-fast bins
        (debug_warn,) = ax_overlay.plot(
            [], [], color='red', lw=2.6, drawstyle='steps-mid',
            label=r'cooling $\tau$ < $\Delta t$', zorder=4)
    else:
        debug_warn = None
    time_text = ax_sed.text(
        0.97, 0.97, '', transform=ax_sed.transAxes,
        ha='right', va='top', fontsize=14,
        bbox=dict(boxstyle='round,pad=0.3', fc='white',
                  ec='k', alpha=0.85),
    )

    if show_data:
        _label_seen = set()
        for _obs in observables:
            if _obs.obs_type != ObsType.FLUX_DENSITY:
                continue
            if _obs._flux_unit not in ('erg/s/cm2', 'nufnu'):
                continue
            _E_eV = _obs._freq_hz * _h_eV
            _color = _color_for_label(_obs.label)
            _root = next(
                (k for k in _DATASET_COLORS if k in _obs.label), None)
            if _root is None:
                _root = (_obs.label.split()[0]
                         if _obs.label else 'data')
            if _root in _label_seen:
                _lbl = ''
            else:
                _lbl = _root
                _label_seen.add(_root)
            if _obs.upper_limit:
                ax_sed.errorbar(_E_eV, _obs.value,
                                yerr=_obs.value * 0.6,
                                uplims=True, c=_color, ls='none',
                                marker='o', ms=4,
                                label=_lbl, zorder=4)
            else:
                ax_sed.errorbar(_E_eV, _obs.value, yerr=_obs.error,
                                c=_color, ls='none', marker='o',
                                ms=4, capsize=2,
                                label=_lbl, zorder=4)

    h1, l1 = ax_sed.get_legend_handles_labels()
    h2, l2 = ax_overlay.get_legend_handles_labels()
    ax_sed.legend(h1 + h2, l1 + l2, loc='upper left', frameon=False)

    circle_snr = circle_pwn = circle_rs = side_text = None
    if side_panel:
        r_snr_max = max(d.rad_snr for d in dyninfo)
        for ref in (reference_radii or []):
            r_snr_max = max(r_snr_max, ref.get('radius', 0.0),
                            ref.get('r_hi', 0.0), ref.get('shade_hi', 0.0))
        half = r_snr_max * 1.15
        ax_side.set_xlim(-half, half)
        ax_side.set_ylim(-half, half)
        ax_side.set_aspect('equal', adjustable='box')
        ax_side.set_xticks([])
        ax_side.set_yticks([])
        for spine in ax_side.spines.values():
            spine.set_visible(False)

        circle_snr = Circle((0, 0), 0.0, fill=True, lw=0,
                            color=SNR_COLOR, zorder=2)
        circle_pwn = Circle((0, 0), 0.0, fill=True, lw=0,
                            color=PWN_COLOR, alpha=0.95, zorder=3)
        circle_rs  = Circle((0, 0), 0.0, fill=False, lw=1.4,
                            ls='-', color='k', zorder=4)
        ax_side.add_patch(circle_snr)
        ax_side.add_patch(circle_pwn)
        ax_side.add_patch(circle_rs)

        swatches = [
            Line2D([0], [0], color=SNR_COLOR, lw=12, label='SNR'),
            Line2D([0], [0], color=PWN_COLOR, lw=12, label='PWN'),
            Line2D([0], [0], color='k', lw=1.4, ls='-', label='RS'),
        ]
        for ref in (reference_radii or []):
            r_color = ref.get('color', 'k')
            r_ls = ref.get('ls', '--')
            r_lw = ref.get('lw', 1.0)
            r_label = ref.get('label', '')
            if 'shade_lo' in ref and 'shade_hi' in ref:
                ax_side.add_patch(Wedge((0, 0), ref['shade_hi'], 0, 360,
                                        width=ref['shade_hi'] - ref['shade_lo'],
                                        fc=r_color, ec='none',
                                        alpha=ref.get('shade_alpha', 0.18),
                                        zorder=4.5))
            if 'radius' in ref:
                ax_side.add_patch(Circle((0, 0), ref['radius'], fill=False,
                                         lw=r_lw, ls=r_ls, color=r_color,
                                         zorder=5))
            else:
                for r in (ref['r_lo'], ref['r_hi']):
                    ax_side.add_patch(Circle((0, 0), r, fill=False,
                                             lw=r_lw, ls=r_ls,
                                             color=r_color, zorder=5))
            if r_label:
                swatches.append(Line2D([0], [0], color=r_color, lw=r_lw,
                                       ls=r_ls, label=r_label))
        ax_side.legend(handles=swatches, loc='lower right',
                       frameon=False, fontsize=9,
                       handlelength=1.8)

        side_text = ax_side.text(
            0.02, 0.98, '', transform=ax_side.transAxes,
            ha='left', va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', fc='white',
                      ec='none', alpha=0.85),
        )

    res_scatter = None
    res_E_eV = res_data = res_err = res_log_E = None
    if residual_panel:
        _res_obs = [
            o for o in observables
            if o.obs_type == ObsType.FLUX_DENSITY
            and not o.upper_limit
            and o.error is not None and o.error > 0
            and o.range_lo is None and o.range_hi is None
            and o._flux_unit in ('erg/s/cm2', 'nufnu')
        ]
        if _res_obs:
            res_E_eV  = np.array([o._freq_hz * _h_eV for o in _res_obs])
            res_data  = np.array([o.value for o in _res_obs])
            res_err   = np.array([o.error for o in _res_obs])
            res_log_E = np.log10(res_E_eV)
            _colors   = [_color_for_label(o.label) for o in _res_obs]
            res_scatter = ax_res.scatter(
                res_E_eV, np.zeros_like(res_E_eV),
                c=_colors, s=18, zorder=5)
        ax_res.set_xscale('log')
        ax_res.set_xlim(photon_xlim)
        ax_res.set_ylim(-5, 5)
        ax_res.set_xlabel('Photon Energy [eV]')
        ax_res.set_ylabel(r'Residual [$\sigma$]')
        ax_res.axhline(0, color='k', lw=0.6, zorder=1)
        ax_res.axhline(2, color='gray', ls='--',
                       lw=0.5, alpha=0.6, zorder=1)
        ax_res.axhline(-2, color='gray', ls='--',
                       lw=0.5, alpha=0.6, zorder=1)
        ax_res.grid(which='major', alpha=0.3)
        ax_sed.set_xlabel('')
        plt.setp(ax_sed.get_xticklabels(), visible=False)
        _align_ylabels_left(fig, [ax_sed, ax_res])

    def _set_artist(triple, x, y, xmin=None, xmax=None):
        ln, fills, style = triple
        ln.set_data(x, y)
        if not fills:
            return
        if len(y) == 0:
            for f in fills:
                f.set_verts([])
            return
        y_arr = np.asarray(y, dtype=float)
        if style == 'shaded_step':
            fills[0].set_verts([_shaded_step_verts(xmin, xmax, y_arr)])
        else:  # faded_step
            top_factors, bot_factors, _ = _faded_offsets
            for f, tf, bf in zip(fills, top_factors, bot_factors):
                f.set_verts(
                    [_step_slab_verts(xmin, xmax, y_arr * tf, y_arr * bf)]
                )

    def _flatten(triple):
        ln, fills, _ = triple
        return (ln, *fills)

    def _init():
        _set_artist(line_sed, [], [])
        _set_artist(line_elec, [], [])
        time_text.set_text('')
        artists = [*_flatten(line_sed), *_flatten(line_elec), time_text]
        if line_esc is not None:
            _set_artist(line_esc, [], [])
            artists.extend(_flatten(line_esc))
        if line_inj is not None:
            _set_artist(line_inj, [], [])
            artists.extend(_flatten(line_inj))
        if debug_warn is not None:
            debug_warn.set_data([], [])
            artists.append(debug_warn)
        if side_panel:
            circle_snr.set_radius(0.0)
            circle_pwn.set_radius(0.0)
            circle_rs.set_radius(0.0)
            side_text.set_text('')
            artists += [circle_snr, circle_pwn,
                        circle_rs, side_text]
        if res_scatter is not None:
            res_scatter.set_offsets(
                np.column_stack([res_E_eV,
                                 np.zeros_like(res_E_eV)]))
            artists.append(res_scatter)
        return tuple(artists)

    def _update(i):
        t_phot, phot_spec = radinfo[i]
        _, elec_spec_raw  = elecinfo[i]
        pmask = phot_spec > 0
        if pmask.any():
            nufnu = (1e40 * (phot_spec[pmask] * freq[pmask])
                     / (4.0 * np.pi * d_cm ** 2))
            _set_artist(line_sed, phot_E_eV[pmask], nufnu,
                        phot_bin_min_eV[pmask], phot_bin_max_eV[pmask])
        else:
            _set_artist(line_sed, [], [])
        elec_spec = elec_spec_raw[interior]
        emask = elec_spec > 0
        if emask.any():
            dnde   = elec_spec[emask] * 1e40 / widths_ev[emask]
            e2dnde = ((e_erg[emask] ** 2)
                      * dnde * (1.0 / _eV_to_erg))
            _set_artist(line_elec, e_ev[emask], e2dnde,
                        elec_bin_min_eV[emask], elec_bin_max_eV[emask])
        else:
            _set_artist(line_elec, [], [])
        if line_esc is not None:
            esc_spec = escinfo[i][1][interior]
            smask = esc_spec > 0
            if smask.any():
                dnde_s   = esc_spec[smask] * 1e40 / widths_ev[smask]
                e2dnde_s = ((e_erg[smask] ** 2)
                            * dnde_s * (1.0 / _eV_to_erg))
                _set_artist(line_esc, e_ev[smask], e2dnde_s,
                            elec_bin_min_eV[smask], elec_bin_max_eV[smask])
            else:
                _set_artist(line_esc, [], [])
        if line_inj is not None:
            inj_now = _unit_inj * _e_cum_40(t_phot)
            inj_int = inj_now[interior]
            imask = inj_int > 0
            if imask.any():
                dnde_i = inj_int[imask] * 1e40 / _eff_widths_ev[imask]
                e2dnde_i = ((e_erg[imask] ** 2)
                            * dnde_i * (1.0 / _eV_to_erg))
                _set_artist(line_inj, e_ev[imask], e2dnde_i,
                            elec_bin_min_eV[imask], elec_bin_max_eV[imask])
            else:
                _set_artist(line_inj, [], [])
        if debug_warn is not None:
            if i == 0 or not emask.any():
                debug_warn.set_data([], [])
            else:
                t_prev = radinfo[i - 1][0]
                dt_yr = t_phot - t_prev
                if dt_yr <= 0:
                    debug_warn.set_data([], [])
                else:
                    B_now  = float(np.interp(t_phot, _dyn_times_yr, _dyn_bpwn_uG))
                    B_prev = float(np.interp(t_prev, _dyn_times_yr, _dyn_bpwn_uG))
                    B_use = max(B_now, B_prev)
                    # tau_cool monotonic in E: bins above E_thr are fast
                    E_thr_tev = 1.0 / (dt_yr * (1.0 / _COOL_C_IC
                                                + (B_use ** 2) / _COOL_C_SYNC))
                    em_fast = _e_tev_per_bin[emask] > E_thr_tev
                    if em_fast.any():
                        red_y = np.where(em_fast, e2dnde, np.nan)
                        debug_warn.set_data(e_ev[emask], red_y)
                    else:
                        debug_warn.set_data([], [])
        time_text.set_text(_format_time_yr(t_phot))
        artists = [*_flatten(line_sed), *_flatten(line_elec), time_text]
        if line_esc is not None:
            artists.extend(_flatten(line_esc))
        if line_inj is not None:
            artists.extend(_flatten(line_inj))
        if debug_warn is not None:
            artists.append(debug_warn)
        if side_panel:
            dyn = dyninfo[i]
            circle_snr.set_radius(dyn.rad_snr)
            circle_pwn.set_radius(dyn.rad_pwn)
            circle_rs.set_radius(dyn.rad_rs)
            side_text.set_text(
                f'$R_{{\\rm SNR}}$ = {dyn.rad_snr:5.2f} pc\n'
                f'$R_{{\\rm RS}}$  = {dyn.rad_rs:5.2f} pc\n'
                f'$R_{{\\rm PWN}}$ = {dyn.rad_pwn:5.2f} pc\n'
                f'$B$         = {dyn.bpwn:5.1f} $\\mu$G\n'
                f'$v_{{\\rm PWN}}$ = {dyn.vpwn:5.0f} km/s'
            )
            artists += [circle_snr, circle_pwn,
                        circle_rs, side_text]
        if res_scatter is not None:
            _valid = phot_spec > 0
            if int(_valid.sum()) >= 2:
                _log_grid = np.log10(
                    1e40 * (phot_spec[_valid] * freq[_valid])
                    / (4.0 * np.pi * d_cm ** 2))
                _log_model = np.interp(
                    res_log_E,
                    np.log10(phot_E_eV[_valid]),
                    _log_grid,
                    left=-np.inf, right=-np.inf)
                _model_at_E = 10.0 ** _log_model
            else:
                _model_at_E = np.zeros_like(res_E_eV)
            _sigma = (res_data - _model_at_E) / res_err
            res_scatter.set_offsets(
                np.column_stack([res_E_eV, _sigma]))
            artists.append(res_scatter)
        return tuple(artists)

    ani = FuncAnimation(fig, _update, init_func=_init,
                        frames=frame_indices, blit=True,
                        cache_frame_data=False)
    parent = os.path.dirname(os.path.abspath(outfile))
    if parent:
        os.makedirs(parent, exist_ok=True)
    ext = os.path.splitext(outfile)[1].lower()
    writer = 'pillow' if ext == '.gif' else 'ffmpeg'
    save_kwargs = {'writer': writer, 'fps': fps, 'dpi': dpi}
    if writer == 'ffmpeg':
        save_kwargs['bitrate'] = bitrate
    try:
        ani.save(outfile, **save_kwargs)
    except FileNotFoundError as err:
        plt.close(fig)
        raise RuntimeError(
            "ffmpeg not found on PATH.  Install via "
            "`brew install ffmpeg` (macOS) or "
            "`apt install ffmpeg` (Linux)."
        ) from err
    plt.close(fig)
    if verbose:
        print(f"Wrote {outfile} "
              f"({len(frame_indices)} frames @ {fps} fps)")
    return outfile
