"""Single-particle synchrotron and IC kernels (cgs)."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import integrate, special
from scipy.interpolate import CubicSpline

from .. import constants as C

__all__ = [
    "MEC2", "R0", "SIGMA_T", "A_RAD", "HBAR",
    "LogTable", "SeedTable",
    "synch_F_table", "synch_G_table", "synch_F", "synch_G", "synch_G_exact",
    "synch_Q_turbulent", "synch_nu_c", "synch_power_per_electron",
    "synch_photon_rate", "synch_loss_rate",
    "ic_bb_F", "ic_blackbody_kernel", "ic_jones_kernel", "ic_seed_kernel",
    "ic_loss_function_iso", "ic_kn_moment_H", "ic_loss_rate_blackbody",
    "ic_loss_rate_seed", "planck_number_density", "T_CMB",
]

# derived constants
MEC2: float = C.m_electron * C.c_light**2              # erg
HBAR: float = C.h_cgs / (2.0 * np.pi)                   # erg s
R0: float = C.e_charge**2 / MEC2                        # cm
SIGMA_T: float = 8.0 * np.pi / 3.0 * R0**2              # cm^2
A_RAD: float = 8.0 * np.pi**5 * C.k_B**4 / (15.0 * C.h_cgs**3 * C.c_light**3)
T_CMB: float = 2.72548                                  # K, Fixsen 2009

_SQRT3 = np.sqrt(3.0)
_SYN_AMP = _SQRT3 * C.e_charge**3 / MEC2               # erg/s/Hz/G
_NUC_COEF = 3.0 * C.e_charge / (4.0 * np.pi * C.m_electron * C.c_light)
_NU0_COEF = (4.0 / 3.0) * C.e_charge / (2.0 * np.pi * C.m_electron * C.c_light)
_TURB_AMP = 4.0 * np.pi**2 * C.e_charge**2 / (3.0 * C.c_light)
# KAK14 prefactor, 1/s
_IC_BB_PREF: float = 2.0 * R0**2 * MEC2**3 / (np.pi * HBAR**3 * C.c_light**2)
_JONES_PREF: float = 2.0 * np.pi * R0**2 * C.c_light    # cm^3/s
SQRT_2_3: float = np.sqrt(2.0 / 3.0)


# lookup table
class LogTable:
    """Linear interpolation on a uniform ln x grid."""

    def __init__(self, lnx: np.ndarray, y: np.ndarray,
                 lo_slope: float | None = None) -> None:
        lnx = np.asarray(lnx, dtype=float)
        self.lnx = lnx
        self.y = np.asarray(y, dtype=float)
        self.lo = float(lnx[0])
        self.h = float(lnx[1] - lnx[0])
        self.inv_h = 1.0 / self.h
        self.n = lnx.size
        self.dy = np.append(np.diff(self.y), 0.0)
        self.lo_slope = lo_slope

    def __call__(self, lnx: np.ndarray) -> np.ndarray:
        lnx = np.asarray(lnx, dtype=float)
        shape = lnx.shape
        u = (lnx.reshape(-1) - self.lo) * self.inv_h
        below = np.minimum(u, 0.0) if self.lo_slope is not None else None
        np.clip(u, 0.0, self.n - 1.0, out=u)
        i = u.astype(np.intp)
        np.minimum(i, self.n - 2, out=i)
        u -= i
        out = self.y[i] + u * self.dy[i]
        if below is not None:
            out += (self.lo_slope * self.h) * below
        return out.reshape(shape)


def _gl_panels(a: np.ndarray, b: np.ndarray, n_panels: int, order: int = 8):
    """Composite Gauss-Legendre nodes and weights."""
    t, w = leggauss(order)
    s = (np.arange(n_panels)[:, None] + 0.5 * (t[None, :] + 1.0)) / n_panels
    s = s.ravel()
    ws = np.tile(w, n_panels) * (0.5 / n_panels)
    a = np.asarray(a, dtype=float)[..., None]
    b = np.asarray(b, dtype=float)[..., None]
    return a + (b - a) * s, (b - a) * ws


# synchrotron
_SYN_LNX_MIN = np.log(1e-10)
_SYN_LNX_MAX = np.log(1e3)
_SYN_PER_DECADE = 1000


def synch_G_exact(x: np.ndarray) -> np.ndarray:
    """Exact G(x), AKP10 D5."""
    x = np.asarray(x, dtype=float)
    k1 = special.kv(1.0 / 3.0, 0.5 * x)
    k2 = special.kv(2.0 / 3.0, 0.5 * x)
    return (x / 20.0) * ((8.0 + 3.0 * x**2) * k1**2
                         + x * k2 * (2.0 * k1 - 3.0 * x * k2))


@lru_cache(maxsize=None)
def synch_G_table() -> LogTable:
    """Cached ln(G e^x) table."""
    n = int(round((_SYN_LNX_MAX - _SYN_LNX_MIN) / np.log(10) * _SYN_PER_DECADE)) + 1
    lnx = np.linspace(_SYN_LNX_MIN, _SYN_LNX_MAX, n)
    x = np.exp(lnx)
    k1 = special.kve(1.0 / 3.0, 0.5 * x)
    k2 = special.kve(2.0 / 3.0, 0.5 * x)
    ge = (x / 20.0) * ((8.0 + 3.0 * x**2) * k1**2
                       + x * k2 * (2.0 * k1 - 3.0 * x * k2))
    return LogTable(lnx, np.log(ge), lo_slope=1.0 / 3.0)


@lru_cache(maxsize=None)
def synch_F_table() -> LogTable:
    """Cached ln(F e^x) table."""
    n = int(round((_SYN_LNX_MAX - _SYN_LNX_MIN) / np.log(10) * _SYN_PER_DECADE)) + 1
    lnx = np.linspace(_SYN_LNX_MIN, _SYN_LNX_MAX, n)
    x = np.exp(lnx)
    u, w = _gl_panels(lnx[:-1], lnx[1:], 1, 8)
    t = np.exp(u)
    # scaled interval integrals
    J = np.sum(w * special.kve(5.0 / 3.0, t) * np.exp(-(t - x[:-1, None])) * t,
               axis=-1)
    xN = x[-1]
    tail, _ = integrate.quad(
        lambda tt: special.kve(5.0 / 3.0, tt) * np.exp(-(tt - xN)),
        xN, np.inf, epsabs=0.0, epsrel=1e-13, limit=200)
    decay = np.exp(-(x[1:] - x[:-1]))
    S = np.empty(n)
    S[-1] = tail
    for i in range(n - 2, -1, -1):
        S[i] = J[i] + decay[i] * S[i + 1]
    return LogTable(lnx, np.log(x * S), lo_slope=1.0 / 3.0)


def _table_eval(table: LogTable, x: np.ndarray) -> np.ndarray:
    """f(x) from a ln(f e^x) table."""
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        lnx = np.log(np.where(x > 0, x, 1.0))
        out = np.exp(table(lnx) - x)
    return np.where(x > 0, out, 0.0)


def synch_F(x: np.ndarray) -> np.ndarray:
    """Synchrotron F(x)."""
    return _table_eval(synch_F_table(), x)


def synch_G(x: np.ndarray) -> np.ndarray:
    """Synchrotron G(x)."""
    return _table_eval(synch_G_table(), x)


def synch_Q_turbulent(x: np.ndarray) -> np.ndarray:
    """Gaussian-turbulence Q(x), DA19 Eq. 10."""
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        x23 = np.cbrt(x) ** 2
        out = (1.0 + 1.0 / x23) * np.exp(-2.0 * x23)
    return np.where(x > 0, out, 0.0)


def synch_nu_c(gamma: np.ndarray, B: float) -> np.ndarray:
    """Critical frequency, Hz."""
    return _NUC_COEF * B * np.asarray(gamma, dtype=float) ** 2


def synch_power_per_electron(nu: np.ndarray, gamma: np.ndarray, B: float,
                             kernel: str = "isotropic",
                             sin_alpha: float = SQRT_2_3) -> np.ndarray:
    """Single-electron P_nu, erg/s/Hz."""
    nu = np.asarray(nu, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    if kernel == "isotropic":
        return _SYN_AMP * B * synch_G(nu / synch_nu_c(gamma, B))
    if kernel == "fixed":
        return (_SYN_AMP * B * sin_alpha
                * synch_F(nu / (synch_nu_c(gamma, B) * sin_alpha)))
    if kernel == "turbulent":
        nu0 = _NU0_COEF * B * gamma**2
        return _TURB_AMP * nu / gamma**2 * synch_Q_turbulent(nu / nu0)
    raise ValueError(f"unknown synchrotron kernel {kernel!r}; "
                     "use 'isotropic', 'fixed' or 'turbulent'")


def synch_photon_rate(eps: np.ndarray, gamma: np.ndarray, B: float,
                      kernel: str = "isotropic",
                      sin_alpha: float = SQRT_2_3) -> np.ndarray:
    """Single-electron photon rate, 1/(s erg)."""
    w, g = _outer(eps, gamma)
    eps = w * MEC2
    nu = eps / C.h_cgs
    inv = 1.0 / (C.h_cgs * eps)
    if kernel == "isotropic":
        x = nu / synch_nu_c(g, B)
        return (_SYN_AMP * B) * _fast_table(synch_G_table(), x) * inv
    if kernel == "fixed":
        x = nu / (synch_nu_c(g, B) * sin_alpha)
        return (_SYN_AMP * B * sin_alpha) * _fast_table(synch_F_table(), x) * inv
    if kernel == "turbulent":
        nu0 = _NU0_COEF * B * g**2
        return (_TURB_AMP / C.h_cgs**2) * synch_Q_turbulent(nu / nu0) / g**2
    raise ValueError(f"unknown synchrotron kernel {kernel!r}")


def _fast_table(table: LogTable, x: np.ndarray) -> np.ndarray:
    """f(x) from table, x > 0."""
    with np.errstate(under="ignore"):
        y = table(np.log(x))
        y -= x
        return np.exp(y, out=y)


def synch_loss_rate(gamma: np.ndarray, B: float) -> np.ndarray:
    """Synchrotron loss rate, erg/s."""
    return (4.0 / 3.0) * SIGMA_T * C.c_light * B**2 / (8.0 * np.pi) \
        * np.asarray(gamma, dtype=float) ** 2


# blackbody IC
_IC_LNX_MIN = np.log(1e-12)
_IC_LNX_MAX = np.log(1e3)


def _ic_bb_exact(x0: np.ndarray, n_panels: int = 240, order: int = 8):
    """Exact F1..F4 times e^x0, KAK14 Eqs. 12-16."""
    x0 = np.asarray(x0, dtype=float)
    lo = np.log(1e-10 * np.minimum(x0, 1.0))
    u, w = _gl_panels(lo, np.full_like(x0, np.log(70.0)), n_panels, order)
    t = np.exp(u)
    xx = x0[..., None] + t
    # scaled Planck factor
    pe = np.exp(-t) / (-np.expm1(-xx))
    wt = w * t * pe
    X0 = x0[..., None]
    f1 = np.sum(wt * xx, axis=-1)
    f2 = np.sum(wt * (t**2 + X0**2) / xx, axis=-1)
    f3 = np.sum(wt * t, axis=-1)
    f4 = np.sum(wt * (t - 2.0 * X0 * np.log1p(t / X0) + 2.0 * X0 * t / xx),
                axis=-1)
    return f1, f2, f3, f4


@lru_cache(maxsize=None)
def _ic_bb_tables() -> tuple[LogTable, LogTable, LogTable, LogTable]:
    """Cached F1..F4 tables."""
    nc = int(round((_IC_LNX_MAX - _IC_LNX_MIN) / np.log(10) * 100)) + 1
    lnc = np.linspace(_IC_LNX_MIN, _IC_LNX_MAX, nc)
    fs = _ic_bb_exact(np.exp(lnc))
    nd = int(round((_IC_LNX_MAX - _IC_LNX_MIN) / np.log(10) * 4000)) + 1
    lnd = np.linspace(_IC_LNX_MIN, _IC_LNX_MAX, nd)
    return tuple(LogTable(lnd, CubicSpline(lnc, np.log(f))(lnd)) for f in fs)


def ic_bb_F(k: int, x0: np.ndarray) -> np.ndarray:
    """Exact KAK14 F_k(x0)."""
    return _table_eval(_ic_bb_tables()[k - 1], x0)


def planck_number_density(eps: np.ndarray, T: float,
                          u: float | None = None) -> np.ndarray:
    """Planck photon density, 1/(cm^3 erg)."""
    eps = np.asarray(eps, dtype=float)
    kappa = 1.0 if u is None else u / (A_RAD * T**4)
    with np.errstate(over="ignore"):
        return kappa * 8.0 * np.pi / (C.h_cgs * C.c_light) ** 3 * eps**2 \
            / np.expm1(eps / (C.k_B * T))


def ic_blackbody_kernel(eps: np.ndarray, gamma: np.ndarray, T: float,
                        u: float | None = None,
                        theta: float | None = None) -> np.ndarray:
    """Blackbody IC rate, KAK14 Eq. 11/14."""
    w, g = _outer(eps, gamma)
    Tm = C.k_B * T / MEC2
    kappa = 1.0 if u is None else u / (A_RAD * T**4)
    z = w / g
    ok = (z < 1.0) & (g > 1.0)
    zz = np.where(ok, z, 0.5)
    tabs = _ic_bb_tables()
    if theta is None:
        tt = 4.0 * g * Tm
        ta, tb = tabs[2], tabs[3]
    else:
        tt = 2.0 * g * Tm * (1.0 - np.cos(theta))
        ta, tb = tabs[0], tabs[1]
    omz = 1.0 - zz
    x0 = zz / (omz * tt)
    with np.errstate(under="ignore", over="ignore"):
        lnx0 = np.log(x0)
        val = (zz * zz / (2.0 * omz)) * np.exp(ta(lnx0)) + np.exp(tb(lnx0))
        val *= np.exp(-x0)
    val *= (_IC_BB_PREF / MEC2) * kappa * Tm**2 / g**2
    return np.where(ok, val, 0.0)


def _outer(eps: np.ndarray, gamma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Broadcast photon and electron arrays."""
    w = (np.asarray(eps, dtype=float).ravel() / MEC2)[None, :]
    g = np.asarray(gamma, dtype=float)
    g = g.ravel()[:, None] if g.ndim < 2 else g
    return w, g


# Jones kernel
def _bg_F(q: np.ndarray, b: np.ndarray) -> np.ndarray:
    """BG70 Eq. 2.48 bracket."""
    bq = b * q
    return (2.0 * q * np.log(q) + (1.0 + 2.0 * q) * (1.0 - q)
            + 0.5 * bq * bq * (1.0 - q) / (1.0 + bq))


def ic_jones_kernel(eps: np.ndarray, gamma: np.ndarray,
                    eps0: np.ndarray) -> np.ndarray:
    """Jones kernel per seed photon."""
    eps = np.asarray(eps, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    eps0 = np.asarray(eps0, dtype=float)
    z = eps / (gamma * MEC2)
    b = 4.0 * gamma * eps0 / MEC2
    with np.errstate(divide="ignore", invalid="ignore"):
        q = z / (b * (1.0 - z))
        ok = (z < 1.0) & (q <= 1.0) & (q >= 0.25 / gamma**2)
        qq = np.where(ok, q, 0.5)
        val = _bg_F(qq, b)
    return np.where(ok, _JONES_PREF * val / (gamma**2 * eps0), 0.0)


class SeedTable:
    """Tabulated isotropic seed spectrum."""

    def __init__(self, eps: np.ndarray, dnde: np.ndarray,
                 interpolation: str = "cubic") -> None:
        eps = np.asarray(eps, dtype=float).ravel()
        dnde = np.asarray(dnde, dtype=float).ravel()
        keep = (dnde > 0) & (eps > 0) & np.isfinite(dnde)
        eps, dnde = eps[keep], dnde[keep]
        order = np.argsort(eps)
        eps, dnde = eps[order], dnde[order]
        if eps.size < 2:
            raise ValueError("seed table needs at least two positive points")
        lne, lnn = np.log(eps), np.log(dnde)
        self.lnw_min = lne[0] - np.log(MEC2)
        self.lnw_max = lne[-1] - np.log(MEC2)
        nd = max(int(np.ceil((lne[-1] - lne[0]) / np.log(10) * 1000)), 16) + 1
        grid = np.linspace(lne[0], lne[-1], nd)
        if interpolation == "cubic" and eps.size >= 4:
            vals = CubicSpline(lne, lnn)(grid)
        elif interpolation in ("cubic", "linear"):
            vals = np.interp(grid, lne, lnn)
        else:
            raise ValueError("interpolation must be 'cubic' or 'linear'")
        self.eps = eps
        self.dnde = dnde
        self._table = LogTable(grid - np.log(MEC2), vals)

    def density(self, eps: np.ndarray) -> np.ndarray:
        """dn/deps, zero off-table."""
        lnw = np.log(np.asarray(eps, dtype=float) / MEC2)
        inside = (lnw >= self.lnw_min) & (lnw <= self.lnw_max)
        return np.where(inside, np.exp(self._table(lnw)), 0.0)

    def _density_lnw(self, lnw: np.ndarray) -> np.ndarray:
        return np.exp(self._table(lnw))

    def energy_density(self) -> float:
        """Total energy density, erg/cm^3."""
        lnw = np.linspace(self.lnw_min, self.lnw_max, 20001)
        e = np.exp(lnw) * MEC2
        return float(np.trapezoid(e * e * self._density_lnw(lnw), lnw))


def ic_seed_kernel(eps: np.ndarray, gamma: np.ndarray, seed: SeedTable,
                   rtol: float = 1e-6, n_panels: int = 4,
                   max_panels: int = 512,
                   chunk: int = 2_000_000) -> np.ndarray:
    """Tabulated-seed IC rate, 1/(s erg)."""
    w, g = _outer(eps, gamma)
    z = w / g
    g = np.broadcast_to(g, z.shape)
    ok = z < 1.0
    zz = np.where(ok, z, 0.5)
    with np.errstate(divide="ignore"):
        lo = np.log(zz / (4.0 * g * (1.0 - zz)))
        hi = np.log(zz * g / (1.0 - zz))
    lo = np.maximum(lo, seed.lnw_min)
    hi = np.minimum(hi, seed.lnw_max)
    ok &= hi > lo
    idx = np.nonzero(ok)
    lo_v, hi_v = lo[idx], hi[idx]
    zv = zz[idx]
    gv = g[idx]

    def integral(m: int) -> np.ndarray:
        out = np.empty(lo_v.size)
        step = max(1, chunk // (m * 8))
        for s in range(0, lo_v.size, step):
            sl = slice(s, s + step)
            u, wt = _gl_panels(lo_v[sl], hi_v[sl], m, 8)
            b = 4.0 * gv[sl, None] * np.exp(u)
            q = zv[sl, None] / (b * (1.0 - zv[sl, None]))
            q = np.minimum(q, 1.0)
            val = seed._density_lnw(u) * _bg_F(q, b)
            out[sl] = np.sum(wt * val, axis=-1)
        return out

    m = n_panels
    prev = integral(m)
    while True:
        m *= 2
        cur = integral(m)
        scale = np.max(np.abs(cur)) if cur.size else 0.0
        sig = np.abs(cur) > 1e-12 * scale
        err = np.max(np.abs(cur - prev)[sig] / np.abs(cur[sig])) if np.any(sig) else 0.0
        if err <= rtol or m >= max_panels:
            break
        prev = cur
    res = np.zeros_like(z)
    res[idx] = cur * _JONES_PREF / gv**2
    return res


# IC loss rates
def _H_exact(b: np.ndarray, n_panels: int = 240) -> np.ndarray:
    """Loss moment H(b) by quadrature."""
    b = np.asarray(b, dtype=float)
    lo = np.log(1e-12 * np.minimum(1.0, 1.0 / b))
    u, w = _gl_panels(lo, np.zeros_like(b), n_panels, 8)
    q = np.exp(u)
    B = b[..., None]
    return np.sum(w * q * B**2 * q * _bg_F(q, B) / (1.0 + B * q) ** 3, axis=-1)


@lru_cache(maxsize=None)
def _H_spline() -> CubicSpline:
    lnb = np.linspace(np.log(1e-12), np.log(1e14), 26 * 20 + 1)
    return CubicSpline(lnb, np.log(_H_exact(np.exp(lnb))))


def ic_kn_moment_H(b: np.ndarray) -> np.ndarray:
    """Loss moment H(b)."""
    b = np.asarray(b, dtype=float)
    spl = _H_spline()
    lnb = np.log(np.clip(b, 1e-300, None))
    out = np.exp(spl(np.clip(lnb, np.log(1e-12), np.log(1e14))))
    small = b < 1e-12
    return np.where(small, b**2 / 9.0, out)


@lru_cache(maxsize=None)
def _Fiso_table() -> LogTable:
    """Cached ln F_iso table."""
    lnt = np.linspace(np.log(1e-8), np.log(1e10), 18 * 25 + 1)
    t = np.exp(lnt)
    u, w = _gl_panels(np.full_like(t, np.log(1e-10)), np.full_like(t, np.log(70.0)),
                      240, 8)
    s = np.exp(u)
    F = np.sum(w * s * s * ic_kn_moment_H(s * t[:, None]) / np.expm1(s), axis=-1)
    lnd = np.linspace(lnt[0], lnt[-1], 18 * 1000 + 1)
    return LogTable(lnd, CubicSpline(lnt, np.log(F))(lnd), lo_slope=2.0)


def ic_loss_function_iso(t: np.ndarray) -> np.ndarray:
    """Exact F_iso(t), KAK14 Eq. 36."""
    t = np.asarray(t, dtype=float)
    return np.exp(_Fiso_table()(np.log(t)))


def ic_loss_rate_blackbody(E: np.ndarray, T: float,
                           u: float | None = None) -> np.ndarray:
    """Blackbody IC loss rate, erg/s."""
    g = np.asarray(E, dtype=float) / MEC2
    Tm = C.k_B * T / MEC2
    kappa = 1.0 if u is None else u / (A_RAD * T**4)
    return _IC_BB_PREF * MEC2 * kappa * Tm**2 * ic_loss_function_iso(4.0 * g * Tm)


def ic_loss_rate_seed(E: np.ndarray, seed: SeedTable,
                      n_nodes: int = 4000) -> np.ndarray:
    """Tabulated-seed IC loss rate, erg/s."""
    g = np.atleast_1d(np.asarray(E, dtype=float) / MEC2)
    u, w = _gl_panels(np.array(seed.lnw_min), np.array(seed.lnw_max),
                      n_nodes // 8, 8)
    n = seed._density_lnw(u)
    H = ic_kn_moment_H(4.0 * g[:, None] * np.exp(u)[None, :])
    out = _JONES_PREF * MEC2**2 * np.sum(w * n * H, axis=-1)
    return out.reshape(np.shape(E)) if np.ndim(E) else out[0]
