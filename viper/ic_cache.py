"""Per-process IC tables (grids and field temperature only)."""

from __future__ import annotations

import copy
import math
from collections import OrderedDict

import numpy as np

from . import constants as C
from .grids import SpectralGrid, LOG_EMPTY, LOG_EMPTY_CUT
from .radiation import (
    calc_sigma_and_pow_nu_ic, ic_cooling_matrix, blackbody_number_density,
)

T_NODE_DEX = 0.02        # node spacing
MAX_FIELD_NODES = 128    # LRU cap


def grid_key(grid: SpectralGrid) -> tuple:
    """Grid geometry key."""
    return (int(grid.n_bins),
            round(float(grid.log_avg[1]), 10),
            round(float(grid.log_avg[grid.n_bins - 2]), 10))


# --- K2 cooling matrix -------------------------------------------------

_K2: dict = {}


def cached_k2(elec: SpectralGrid, phot: SpectralGrid,
              target: SpectralGrid) -> np.ndarray:
    """K2 matrix, built once per grid triple."""
    key = (grid_key(elec), grid_key(phot), grid_key(target))
    arr = _K2.get(key)
    if arr is None:
        arr = ic_cooling_matrix(elec, phot, target)
        arr.setflags(write=False)
        _K2[key] = arr
    return arr


# --- blackbody field kernels -------------------------------------------

_NODES: "OrderedDict[tuple, np.ndarray]" = OrderedDict()


def fill_blackbody(target: SpectralGrid, temp: float, norm_dim: float) -> None:
    """Blackbody target field (same bins as _fill_target_field)."""
    n = target.n_bins - 2
    target.number[:] = 0.0
    for i in range(n):
        target.number[i] = norm_dim * blackbody_number_density(
            target.bin_avg[i], temp)
    target.update_log_number()


def norm_dim_from_u(temp: float, u_evcm3: float) -> float:
    """eV/cm^3 -> blackbody multiplier."""
    if temp > 0.0 and u_evcm3 > 0.0:
        return u_evcm3 / (C.radiation_const * temp**4 * C.eV_per_erg)
    return 0.0


def node_temperature(n: int) -> float:
    return 10.0 ** (n * T_NODE_DEX)


def _node_kernel(n: int, elec, phot, target, flags) -> np.ndarray:
    """Unit blackbody kernel at node n."""
    key = (grid_key(elec), grid_key(phot), grid_key(target), int(n), flags)
    arr = _NODES.get(key)
    if arr is not None:
        _NODES.move_to_end(key)
        return arr
    tgt = copy.deepcopy(target)
    fill_blackbody(tgt, node_temperature(n), 1.0)
    arr = calc_sigma_and_pow_nu_ic(
        elec, phot, tgt,
        ic_early_break=flags[0], ic_j_break=flags[1],
        ic_skip_empty_bins=flags[2])
    arr.setflags(write=False)
    _NODES[key] = arr
    while len(_NODES) > MAX_FIELD_NODES:
        _NODES.popitem(last=False)
    return arr


def _blend(lo: np.ndarray, hi: np.ndarray, frac: float) -> np.ndarray:
    """Interpolate two log kernels."""
    both = (lo > LOG_EMPTY_CUT) & (hi > LOG_EMPTY_CUT)
    plo = np.where(lo > LOG_EMPTY_CUT, 10.0 ** lo, 0.0)
    phi = np.where(hi > LOG_EMPTY_CUT, 10.0 ** hi, 0.0)
    lin = (1.0 - frac) * plo + frac * phi
    out = np.where(lin > 0.0, np.log10(np.maximum(lin, 1e-300)), LOG_EMPTY)
    return np.where(both, lo + frac * (hi - lo), out)


def field_kernel(temp: float, u_evcm3: float, elec, phot, target,
                 ic_early_break=True, ic_j_break=True,
                 ic_skip_empty_bins=True) -> np.ndarray:
    """Kernel of one field."""
    norm_dim = norm_dim_from_u(temp, u_evcm3)
    if norm_dim <= 0.0:
        return np.full((elec.n_bins, phot.n_bins), LOG_EMPTY)
    flags = (bool(ic_early_break), bool(ic_j_break), bool(ic_skip_empty_bins))
    x = math.log10(temp) / T_NODE_DEX
    n_lo = math.floor(x)
    frac = x - n_lo
    if frac < 1e-9:
        k = np.array(_node_kernel(n_lo, elec, phot, target, flags))
    elif frac > 1.0 - 1e-9:
        k = np.array(_node_kernel(n_lo + 1, elec, phot, target, flags))
    else:
        k = _blend(_node_kernel(n_lo, elec, phot, target, flags),
                   _node_kernel(n_lo + 1, elec, phot, target, flags), frac)
    shift = math.log10(norm_dim)
    return np.where(k > LOG_EMPTY_CUT, k + shift, LOG_EMPTY)


def log_sum(kernels) -> np.ndarray:
    """Sum of log kernels."""
    lin = None
    for k in kernels:
        p = np.where(k > LOG_EMPTY_CUT, 10.0 ** k, 0.0)
        lin = p if lin is None else lin + p
    return np.where(lin > 0.0, np.log10(np.maximum(lin, 1e-300)), LOG_EMPTY)


def external_field_kernel(cmb_kernel: np.ndarray, elec, phot, target,
                          ic_temps, ic_norms, ic_early_break=True,
                          ic_j_break=True, ic_skip_empty_bins=True) -> np.ndarray:
    """CMB table + field kernels."""
    ks = [cmb_kernel]
    for T, u in zip(ic_temps, ic_norms):
        ks.append(field_kernel(float(T), float(u), elec, phot, target,
                               ic_early_break, ic_j_break, ic_skip_empty_bins))
    return log_sum(ks)


def clear() -> None:
    """Empty the caches."""
    _K2.clear()
    _NODES.clear()
