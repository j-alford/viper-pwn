"""Closed forms and harnesses for the analytic-limit tests."""
import numpy as np

import viper.constants as C
from viper.grids import make_log_grid
from viper.evolution import evolve_spectrum


# --- synchrotron cooling -----------------------------------------------

def sigma_thomson():
    """8 pi e^4 / (3 m^2 c^4)."""
    e, m, c = C.e_charge, C.m_electron, C.c_light
    return 8.0 * np.pi * e**4 / (3.0 * m**2 * c**4)


def synch_b_coeff(b_gauss):
    """(4/3) sigma_T c U_B / (m c^2)^2  [1/s/erg]."""
    u_b = b_gauss**2 / (8.0 * np.pi)
    mc2 = C.m_electron * C.c_light**2
    return 4.0 / 3.0 * sigma_thomson() * C.c_light * u_b / mc2**2


# --- piecewise power law in bin counts ----------------------------------

def powerlaw_counts(edges_lo, edges_hi, k, p, e_lo, e_hi):
    """Integral of k E^-p over [lo, hi] clipped to [e_lo, e_hi]."""
    lo = np.clip(edges_lo, e_lo, e_hi)
    hi = np.clip(edges_hi, e_lo, e_hi)
    with np.errstate(divide='ignore', invalid='ignore'):
        if p == 1.0:
            out = k * np.log(hi / lo)
        else:
            out = k * (hi ** (1.0 - p) - lo ** (1.0 - p)) / (1.0 - p)
    return np.where(hi > lo, out, 0.0)


def powerlaw_norm(energy_erg, p, e_lo, e_hi):
    """k such that integral E k E^-p dE over [e_lo, e_hi] (GeV) is energy_erg."""
    if p == 2.0:
        integ = np.log(e_hi / e_lo)
    else:
        integ = (e_hi ** (2.0 - p) - e_lo ** (2.0 - p)) / (2.0 - p)
    return energy_erg * C.GeV_per_erg / integ


# --- cooling harness ----------------------------------------------------

class CoolingHarness:
    """evolve_spectrum at fixed B and dt; E_N R_N = E_0 R_0 / (1 + b E_0 R_0 Theta)."""

    def __init__(self, n_bins, b_gauss, dt_s, e_grid_lo=1e-6, e_grid_hi=1e8,
                 r0_cm=C.pc, eps=1e-6, n_elec_40=1e-30, pow_ic=None):
        self.grid = make_log_grid(e_grid_lo, e_grid_hi, n_bins)
        self.b = b_gauss
        self.dt = dt_s
        self.eps = eps
        self.r0 = r0_cm
        self.r = r0_cm
        self.theta = 0.0
        self.n_elec_40 = n_elec_40
        self.pow_ic = (np.zeros(self.grid.n_bins) if pow_ic is None
                       else np.asarray(pow_ic, float))
        self.min_ic_bin = 0 if pow_ic is not None else self.grid.n_bins
        self.cur = self.grid.copy()
        self.cur.reset_number(0.0)
        self.last = None

    @property
    def n_bins(self):
        return self.grid.n_bins

    def set_counts(self, counts):
        self.cur.number[:] = counts
        self.cur.update_log_number()

    def step(self, inject=None):
        r_new = self.r * (1.0 + self.eps)
        v = self.eps * self.r / self.dt
        nxt = self.grid.copy()
        nxt.reset_number(0.0)
        self.last = evolve_spectrum(
            self.cur, nxt, self.b, self.r, r_new, v,
            self.n_elec_40, self.min_ic_bin, self.pow_ic)
        if inject is not None:
            nxt.number += inject
            nxt.update_log_number()
        self.theta += np.log(r_new / self.r) * self.dt / (self.eps * self.r)
        self.r = r_new
        self.cur = nxt

    def run(self, n_steps, inject=None):
        for _ in range(n_steps):
            self.step(inject)

    def rad_rat(self):
        return self.r0 / self.r

    def preimage(self, e_gev, b_coeff):
        """Initial energy of a particle now at e_gev; inf if from infinity."""
        y = e_gev * self.r
        y_erg = y / C.GeV_per_erg
        denom = 1.0 - b_coeff * y_erg * self.theta
        with np.errstate(divide='ignore', invalid='ignore'):
            y0 = np.where(denom > 0, y / denom, np.inf)
        return y0 / self.r0

    def analytic_counts(self, k, p, e_lo, e_hi, b_coeff):
        """Bin counts now, for an initial k E^-p on [e_lo, e_hi]."""
        g = self.grid
        lo0 = self.preimage(g.bin_min, b_coeff)
        hi0 = self.preimage(g.bin_max, b_coeff)
        return powerlaw_counts(lo0, hi0, k, p, e_lo, e_hi)


def steady_state_counts(grid, k, p, e_lo, e_hi, b_coeff):
    """Bin counts of N_ss(E) = k (E^(1-p) - e_hi^(1-p)) / ((p-1) b E^2)."""
    b_gev = b_coeff / C.GeV_per_erg
    ok = grid.bin_min > 0
    lo = np.minimum(np.where(ok, grid.bin_min, e_hi), e_hi)
    hi = np.minimum(grid.bin_max, e_hi)
    a = k / ((p - 1.0) * b_gev)
    t1 = (hi ** (-p) - lo ** (-p)) / (-p)
    t2 = e_hi ** (1.0 - p) * (1.0 / lo - 1.0 / hi)
    out = a * (t1 - t2)
    return np.where(ok & (hi > lo), out, 0.0)


def translation_excess(p, delta_dex, ln_shift, n_steps):
    """(a_num / f^(1-p))^n - 1 for the uniform-in-bin shift scheme."""
    f = np.exp(ln_shift / n_steps)
    q = (f - 1.0) / (10.0 ** delta_dex - 1.0)
    a_num = 1.0 - q * (1.0 - 10.0 ** ((1.0 - p) * delta_dex))
    a_true = f ** (1.0 - p)
    return (a_num / a_true) ** n_steps - 1.0


# --- fitting helpers -----------------------------------------------------

def logslope(x, y, lo, hi):
    m = (x >= lo) & (x <= hi) & (y > 0)
    return np.polyfit(np.log10(x[m]), np.log10(y[m]), 1)[0]


def band_ratio(grid, num, ana, lo, hi):
    """Sum(num)/sum(ana) over bins whose centre lies in [lo, hi]."""
    s = (grid.bin_avg >= lo) & (grid.bin_avg <= hi)
    return num[s].sum() / ana[s].sum()


def bins_per_decade(grid):
    return 1.0 / (grid.log_max[2] - grid.log_max[1])
