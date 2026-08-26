"""Spectral energy grids for electron and photon spectra."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

# Empty-bin log10 sentinels; never change.
LOG_EMPTY = -9999.0
LOG_EMPTY_CUT = -9998.0


@dataclass
class SpectralGrid:
    """Log-spaced spectral bin structure.

    All arrays have length ``n_bins``, including the underflow bin at
    index 0 and the overflow bin at index ``n_bins - 1``.

    Attributes
    ----------
    bin_min, bin_max, bin_avg, bin_width : ndarray
        Physical bin boundaries, geometric average, and width.
    log_min, log_max, log_avg, log_width : ndarray
        Base-10 logarithms of the above (``LOG_EMPTY`` where undefined).
    number : ndarray
        Particle count or luminosity density stored in each bin.
    log_number : ndarray
        Base-10 log of ``number`` (``LOG_EMPTY`` where number <= 0).
    """

    bin_min: np.ndarray
    bin_max: np.ndarray
    bin_avg: np.ndarray
    bin_width: np.ndarray
    log_min: np.ndarray
    log_max: np.ndarray
    log_avg: np.ndarray
    log_width: np.ndarray
    number: np.ndarray
    log_number: np.ndarray

    @property
    def n_bins(self) -> int:
        return len(self.number)

    def reset_number(self, value: float = 0.0) -> None:
        """Reset all bin counts to *value*."""
        self.number[:] = value
        if value > 0:
            self.log_number[:] = np.log10(value)
        else:
            self.log_number[:] = LOG_EMPTY

    def update_log_number(self) -> None:
        """Recompute log_number from number (handling zeros)."""
        mask = self.number > 0
        self.log_number[:] = LOG_EMPTY
        self.log_number[mask] = np.log10(self.number[mask])

    def copy(self) -> SpectralGrid:
        """Return a deep copy."""
        return SpectralGrid(
            bin_min=self.bin_min.copy(),
            bin_max=self.bin_max.copy(),
            bin_avg=self.bin_avg.copy(),
            bin_width=self.bin_width.copy(),
            log_min=self.log_min.copy(),
            log_max=self.log_max.copy(),
            log_avg=self.log_avg.copy(),
            log_width=self.log_width.copy(),
            number=self.number.copy(),
            log_number=self.log_number.copy(),
        )


def make_log_grid(val_min: float, val_max: float, n_bins: int) -> SpectralGrid:
    """Create a logarithmically spaced spectral grid.

    Parameters
    ----------
    val_min : float
        Minimum physical value (e.g. energy in GeV or frequency in Hz).
    val_max : float
        Maximum physical value.
    n_bins : int
        Total number of bins including the two overflow bins.

    Returns
    -------
    SpectralGrid
    """
    log_min_val = np.log10(val_min)
    log_bin_iter = (np.log10(val_max) - log_min_val) / (n_bins - 1)

    b_min = np.zeros(n_bins)
    b_max = np.zeros(n_bins)
    b_avg = np.zeros(n_bins)
    b_width = np.zeros(n_bins)
    b_logmin = np.full(n_bins, LOG_EMPTY)
    b_logmax = np.zeros(n_bins)
    b_logavg = np.zeros(n_bins)
    b_logwidth = np.zeros(n_bins)

    # --- Bin 0: underflow ---
    b_min[0] = 0.0
    b_max[0] = val_min
    b_width[0] = val_min
    b_avg[0] = val_min / 2.0
    b_logmin[0] = LOG_EMPTY
    b_logmax[0] = log_min_val
    b_logavg[0] = log_min_val - 0.5 * log_bin_iter
    b_logwidth[0] = log_min_val

    # --- Interior bins 1 .. n_bins-2 ---
    for i in range(1, n_bins - 1):
        b_min[i] = b_max[i - 1]
        b_max[i] = 10.0 ** (log_min_val + i * log_bin_iter)
        b_width[i] = b_max[i] - b_min[i]
        b_avg[i] = 10.0 ** (log_min_val + (i - 0.5) * log_bin_iter)
        b_logmin[i] = b_logmax[i - 1]
        b_logmax[i] = log_min_val + i * log_bin_iter
        b_logwidth[i] = np.log10(b_width[i])
        b_logavg[i] = log_min_val + (i - 0.5) * log_bin_iter

    # --- Bin n_bins-1: overflow ---
    last = n_bins - 1
    b_min[last] = b_max[last - 1]
    b_max[last] = 100.0 * b_min[last]
    b_width[last] = b_max[last] - b_min[last]
    b_avg[last] = 10.0 ** (log_min_val + (last - 0.5) * log_bin_iter)
    b_logmin[last] = b_logmax[last - 1]
    b_logmax[last] = b_logmax[last - 1] + 2
    b_logwidth[last] = np.log10(b_width[last])
    b_logavg[last] = log_min_val + (last - 0.5) * log_bin_iter

    number = np.zeros(n_bins)
    log_number = np.full(n_bins, LOG_EMPTY)

    return SpectralGrid(
        bin_min=b_min,
        bin_max=b_max,
        bin_avg=b_avg,
        bin_width=b_width,
        log_min=b_logmin,
        log_max=b_logmax,
        log_avg=b_logavg,
        log_width=b_logwidth,
        number=number,
        log_number=log_number,
    )
