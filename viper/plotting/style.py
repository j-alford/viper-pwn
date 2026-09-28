"""Shared style constants and low-level drawing helpers."""

from __future__ import annotations

import numpy as np

from .. import constants as C



# ===================================================================
# Constants (from the package constants module; never hardcode)
# ===================================================================

_eV_to_erg = 1.0 / C.eV_per_erg
_keV_to_erg = 1.0e3 / C.eV_per_erg
_h_eV = C.h_eV  # Planck constant in eV·s

_BIN_STYLE_CHOICES = ('curve', 'step', 'shaded_step', 'faded_step')

_SHADED_STEP_FILL_ALPHA = 0.18
_SHADED_STEP_FLOOR = 1e-300  # below any log-axis ylim

_FADED_STEP_K = 12             # slabs to approximate the gradient
_FADED_STEP_BASE_ALPHA = 0.35  # top-of-stack alpha
_FADED_STEP_DEFAULT_DEX = 1.0  # decades until fade reaches zero


def _apply_sed_style():
    """Apply the shared CMU-Serif mathtext style used by SED plots.

    Mutates global matplotlib rcParams; every public plotter exposes
    an ``apply_style=False`` opt-out."""
    import matplotlib as mpl
    mpl.rcParams['text.usetex'] = False
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['CMU Serif', 'Computer Modern Roman',
                                   'DejaVu Serif']
    mpl.rcParams['mathtext.fontset'] = 'cm'


def _particle_bin_edges(grid, interior, e_max_inj=None):
    """Interior electron-bin widths and upper edges [grid units].

    The bin that straddles the injection cutoff ``e_max_inj`` is
    truncated at ``e_max_inj``: the injection fills only its lower part,
    so dividing its count by the full width under-states the density
    (and its upper edge would overstate the spectrum's extent).
    """
    bmin = grid.bin_min[interior]
    bmax = grid.bin_max[interior].copy()
    if e_max_inj is not None:
        at = (bmin < e_max_inj) & (bmax > e_max_inj)
        bmax[at] = e_max_inj
    return bmax - bmin, bmax


def _validate_bin_style(bin_style):
    """Raise ValueError on anything outside the enumerated set."""
    if bin_style not in _BIN_STYLE_CHOICES:
        raise ValueError(
            f"bin_style must be one of {_BIN_STYLE_CHOICES!r}, "
            f"got {bin_style!r}"
        )


def _shaded_step_verts(xmin, xmax, y, floor=_SHADED_STEP_FLOOR):
    """Polygon following step bin edges down to a finite floor."""
    xmin = np.asarray(xmin, dtype=float)
    xmax = np.asarray(xmax, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n == 0:
        return np.empty((0, 2))
    verts = np.empty((2 * n + 2, 2))
    verts[0]  = (xmin[0],  floor)
    verts[-1] = (xmax[-1], floor)
    verts[1:-1:2, 0] = xmin
    verts[1:-1:2, 1] = y
    verts[2:-1:2, 0] = xmax
    verts[2:-1:2, 1] = y
    return verts


def _step_slab_verts(xmin, xmax, y_top, y_bot):
    """Polygon between two step curves sharing the same bin edges."""
    xmin = np.asarray(xmin, dtype=float)
    xmax = np.asarray(xmax, dtype=float)
    y_top = np.asarray(y_top, dtype=float)
    y_bot = np.asarray(y_bot, dtype=float)
    n = len(y_top)
    if n == 0:
        return np.empty((0, 2))
    verts = np.empty((4 * n, 2))
    verts[0:2*n:2, 0] = xmin
    verts[0:2*n:2, 1] = y_top
    verts[1:2*n:2, 0] = xmax
    verts[1:2*n:2, 1] = y_top
    verts[2*n:4*n:2, 0] = xmax[::-1]
    verts[2*n:4*n:2, 1] = y_bot[::-1]
    verts[2*n + 1:4*n:2, 0] = xmin[::-1]
    verts[2*n + 1:4*n:2, 1] = y_bot[::-1]
    return verts


# Escaped/halo particle population color.
_ESCAPED_COLOR = 'xkcd:orange'


def _faded_step_offsets(fade_dex, K=_FADED_STEP_K):
    """Return (top_factors, bot_factors, alphas) for K stacked slabs."""
    edges = np.linspace(0.0, fade_dex, K + 1)  # slab-edge depths in dex
    top_factors = 10.0 ** (-edges[:-1])
    bot_factors = 10.0 ** (-edges[1:])
    # alpha at slab center
    alphas = _FADED_STEP_BASE_ALPHA * (
        1.0 - (np.arange(K) + 0.5) / K)
    return top_factors, bot_factors, alphas


def _plot_binned(ax, bin_style, x, y, xmin, xmax,
                 *, fade_dex=_FADED_STEP_DEFAULT_DEX, **kwargs):
    """Plot a model curve according to bin_style.

    Returns the Line2D for the curve; 'shaded_step'/'faded_step' also
    add PolyCollections to the axes as a side effect.
    """
    _validate_bin_style(bin_style)
    if bin_style == 'curve':
        (line,) = ax.plot(x, y, **kwargs)
        return line
    kw = dict(kwargs)
    kw['drawstyle'] = 'steps-mid'
    (line,) = ax.plot(x, y, **kw)
    if bin_style in ('shaded_step', 'faded_step'):
        from matplotlib.collections import PolyCollection
        color = kw.get('color', line.get_color())
        zorder = kw.get('zorder', 2) - 0.5
        y_arr = np.asarray(y, dtype=float)
        if bin_style == 'shaded_step':
            poly = PolyCollection(
                [_shaded_step_verts(xmin, xmax, y_arr)],
                alpha=_SHADED_STEP_FILL_ALPHA,
                facecolor=color, edgecolor='none', zorder=zorder,
            )
            ax.add_collection(poly)
        else:
            top_factors, bot_factors, alphas = _faded_step_offsets(fade_dex)
            for tf, bf, a in zip(top_factors, bot_factors, alphas):
                if a <= 0:
                    continue
                poly = PolyCollection(
                    [_step_slab_verts(xmin, xmax, y_arr * tf, y_arr * bf)],
                    alpha=a, facecolor=color, edgecolor='none',
                    zorder=zorder,
                )
                ax.add_collection(poly)
    return line


def _align_ylabels_left(fig, axes):
    """Align the visual *left* edges of rotated ylabels across `axes`."""
    fig.canvas.draw()
    bboxes = [ax.yaxis.label.get_window_extent() for ax in axes]
    target_x0 = min(b.x0 for b in bboxes)
    for ax, bbox in zip(axes, bboxes):
        new_anchor_disp = target_x0 + bbox.width
        new_x_axes = ax.transAxes.inverted().transform(
            (new_anchor_disp, 0))[0]
        ax.yaxis.set_label_coords(new_x_axes, 0.5)
    fig.canvas.draw()
