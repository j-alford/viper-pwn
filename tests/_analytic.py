"""Closed forms and harnesses for the analytic-limit tests."""
import numpy as np
from scipy.integrate import quad
from scipy.special import gamma as gamma_fn, kv, zeta

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


# --- radiation closed forms ----------------------------------------------

def mc2():
    return C.m_electron * C.c_light ** 2


def powerlaw_bin_counts(grid, k, p, e_lo_erg, e_hi_erg):
    """Counts (1e40 units) of k E^-p (E in erg) per bin of a GeV grid."""
    lo = grid.bin_min / C.GeV_per_erg
    hi = grid.bin_max / C.GeV_per_erg
    n = powerlaw_counts(lo, hi, k, p, e_lo_erg, e_hi_erg)
    n[grid.bin_min <= 0] = 0.0
    return n / 1e40


def synch_fx_quad(x):
    """F(x) = x int_x^inf K_5/3."""
    return x * quad(lambda t: kv(5.0 / 3.0, t), x, np.inf)[0]


def synch_powerlaw_integral(p):
    """int x^((p-3)/2) F(x) dx, Gamma form (RL 6.36)."""
    return (2.0 ** ((p + 1) / 2) / (p + 1)
            * gamma_fn(p / 4 + 19 / 12) * gamma_fn(p / 4 - 1 / 12))


def pitch_angle_factor(p, kind):
    """<sin a^((p+1)/2)> isotropic, or the fixed-angle kernel factors."""
    s = np.sqrt(2.0 / 3.0)
    if kind == 'isotropic':
        return np.sqrt(np.pi) / 2 * gamma_fn((p + 5) / 4) / gamma_fn((p + 7) / 4)
    if kind == 'rms_pitch':
        return s ** ((p + 1) / 2)
    if kind == 'legacy':
        return 2.0 / 3.0 * s ** ((p - 1) / 2)
    raise ValueError(kind)


def synch_nu_c(e_erg, b_gauss):
    """3 e B gamma^2 / (4 pi m c), sin a = 1."""
    a = 3.0 * C.e_charge * b_gauss / (4.0 * np.pi * C.m_electron * C.c_light)
    return a * (e_erg / mc2()) ** 2


def synch_powerlaw_lnu(nu, k, p, b_gauss, kind='isotropic'):
    """L_nu [erg/s/Hz] of N(E) = k E^-p (E erg) in a uniform field."""
    c_gamma = k * mc2() ** (1.0 - p)
    amp = np.sqrt(3.0) * C.e_charge ** 3 * b_gauss / mc2()
    a = synch_nu_c(mc2(), b_gauss)
    return (c_gamma * amp * 0.5 * (nu / a) ** (-(p - 1) / 2)
            * synch_powerlaw_integral(p) * pitch_angle_factor(p, kind))


def planck_number_density(eps_erg, temp):
    """Photons per cm^3 per erg."""
    return (8.0 * np.pi / (C.h_cgs * C.c_light) ** 3 * eps_erg ** 2
            / np.expm1(eps_erg / (C.k_B * temp)))


def thomson_loss_rate(e_erg, u_erg):
    """(4/3) sigma_T c gamma^2 u."""
    return 4.0 / 3.0 * C.sigma_T * C.c_light * (e_erg / mc2()) ** 2 * u_erg


def ic_thomson_J(p):
    """int_0^1 q^((p-1)/2) (2 q ln q + 1 + q - 2 q^2) dq."""
    return 4.0 * (p * p + 4 * p + 11) / ((p + 1) * (p + 3) ** 2 * (p + 5))


def ic_thomson_blackbody_lnu(nu, k, p, temp):
    """L_nu [erg/s/Hz] of N(E) = k E^-p on a blackbody, Thomson limit."""
    c_gamma = k * mc2() ** (1.0 - p)
    e1 = C.h_cgs * nu
    kt = C.k_B * temp
    rate = (c_gamma * 3.0 * C.sigma_T * C.c_light / 8.0 * 4.0 ** ((p + 1) / 2)
            * ic_thomson_J(p) * 8.0 * np.pi / (C.h_cgs * C.c_light) ** 3
            * kt ** ((p + 5) / 2) * gamma_fn((p + 5) / 2) * zeta((p + 5) / 2)
            * e1 ** (-(p + 1) / 2))
    return e1 * rate * C.h_cgs


def jones_power_per_photon(gamma, eps0):
    """Energy loss rate per unit target density, BG70 eq. 2.48 [erg cm^3/s]."""
    big = 4.0 * eps0 * gamma / mc2()

    def integrand(q):
        e1 = big * q * gamma * mc2() / (1.0 + big * q)
        de1 = big * gamma * mc2() / (1.0 + big * q) ** 2
        f = (2.0 * q * np.log(q) + (1.0 + 2.0 * q) * (1.0 - q)
             + 0.5 * (big * q) ** 2 * (1.0 - q) / (1.0 + big * q))
        return e1 * f * de1

    val = quad(integrand, 0.25 / gamma ** 2, 1.0, limit=200, epsrel=1e-10)[0]
    return 3.0 * C.sigma_T * C.c_light / (4.0 * gamma ** 2 * eps0) * val


def jones_loss_rate(e_erg, temp, u_erg=None):
    """IC loss rate on a blackbody by quadrature [erg/s]."""
    g = e_erg / mc2()
    kt = C.k_B * temp
    scale = 1.0 if u_erg is None else u_erg / (C.radiation_const * temp ** 4)
    val = quad(lambda x: planck_number_density(x * kt, temp) * kt
               * jones_power_per_photon(g, x * kt), 1e-3, 60.0,
               limit=200, epsrel=1e-9)[0]
    return scale * val


# --- timestep ladders ----------------------------------------------------

def loguniform_dts(t0, age, h):
    """Steps of constant dt/t = h from t0 to age."""
    ts = [t0]
    while ts[-1] * (1.0 + h) < age:
        ts.append(ts[-1] * (1.0 + h))
    ts.append(age)
    return list(np.diff(ts))


def refine_dts(dts, k):
    """Every dt split k ways, sum preserved exactly."""
    seq = [d / k for d in dts for _ in range(k)]
    seq[-1] += sum(dts) - sum(seq)
    return seq


def pair_orders(values):
    """log2 of successive difference ratios along a halving ladder."""
    v = np.asarray(values, float)
    d = np.abs(np.diff(v))
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.log2(d[:-1] / d[1:])
