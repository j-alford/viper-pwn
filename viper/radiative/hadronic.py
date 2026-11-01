"""Bremsstrahlung and p-p cross sections (cgs)."""

from __future__ import annotations

import numpy as np

from .. import constants as C
from . import kernels as K

__all__ = [
    "brems_ee_sigma", "brems_ep_sigma", "brems_ee_eps_max",
    "pp_inelastic_sigma", "pp_pi0_sigma", "pp_pi0_multiplicity",
    "pp_nuclear_enhancement", "pion_gamma_dsigma", "pion_gamma_amax",
    "pion_gamma_emax", "pion_tp_min", "MPC2", "M_PI0", "TP_TH", "ALPHA_FS",
]

# constants
MEC2: float = K.MEC2
R0: float = K.R0
ALPHA_FS: float = C.e_charge**2 / (K.HBAR * C.c_light)
MPC2: float = C.m_proton * C.c_light**2                 # erg
_GEV: float = 1.0 / C.GeV_per_erg                       # erg
_MB: float = 1e-27                                      # cm^2
_MP: float = MPC2 / _GEV                                # GeV
M_PI0: float = 0.1349766                                # PDG14, GeV
_TTH: float = 2.0 * M_PI0 + M_PI0**2 / (2.0 * _MP)      # GeV
TP_TH: float = _TTH * _GEV                              # erg
_T_SWITCH: float = 2e-3 * _GEV / MEC2                 # Baring99 2 MeV

# Baring99 A3 small-eps series
_A3_SERIES = np.array([28 / 3, -70 / 3, 128 / 15, -272 / 15, -1504 / 105,
                       -1712 / 105, -6784 / 315, -9728 / 315, -161792 / 3465,
                       -253952 / 3465, -761344 / 6435, -8794112 / 45045])


# e-e bremsstrahlung
def brems_ee_eps_max(gamma: np.ndarray) -> np.ndarray:
    """Exact e-e photon endpoint, m_e c^2 units."""
    T = np.asarray(gamma, dtype=float) - 1.0
    return 0.5 * T * (1.0 + np.sqrt(T / (T + 2.0)))


def _ee_T_min(w: np.ndarray) -> np.ndarray:
    """Inverse of brems_ee_eps_max."""
    return 4.0 * w / (2.0 - w + np.sqrt(w * (w + 4.0)))


def _a3_bracket(w: np.ndarray) -> np.ndarray:
    """Baring99 A3 eps <= 1/2 polynomial."""
    small = w < 0.02
    ws = np.where(small, w, 0.0)
    ser = np.polyval(_A3_SERIES[::-1], ws)
    wl = np.where(small, 0.25, np.minimum(w, 0.5))
    with np.errstate(divide="ignore", invalid="ignore"):
        lg = np.where(wl < 0.5, np.log1p(-2.0 * wl), 0.0)
        direct = (-1.0 / wl**2 + 3.0 / wl - 4.0 - 4.0 * wl - 8.0 * wl**2
                  - 2.0 * (1.0 - 2.0 * wl) * lg
                  * (0.25 / wl**3 - 0.5 / wl**2 + 3.0 / wl - 2.0 + 4.0 * wl))
    return np.where(small, ser, direct)


def _ee_sigma_rel(w: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Baring99 A1-A4, units r0^2."""
    y = w / g
    s1 = (4.0 * ALPHA_FS / w) * (1.0 + (1.0 / 3.0 - y) * (1.0 - y)) \
        * (np.log(2.0 * g * (g - w) / w) - 0.5)
    lo = (16.0 * (1.0 - w + w * w) * np.log(g / w) + _a3_bracket(w))
    wh = np.maximum(w, 0.5)
    hi = (2.0 / wh) * ((4.0 - 1.0 / wh + 0.25 / wh**2) * np.log(2.0 * g)
                       - 2.0 + 2.0 / wh - 0.625 / wh**2)
    s2 = (ALPHA_FS / (3.0 * w)) * np.where(w <= 0.5, lo, hi)
    A = 1.0 - (8.0 / 3.0) * (g - 1.0) ** 0.2 / (g + 1.0) * np.cbrt(y)
    return (s1 + s2) * A


def _ee_sigma_nr(w: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Baring99 A5-A7, units r0^2."""
    p2 = g * g - 1.0
    x = np.minimum(4.0 * w / p2, 1.0)
    gb = np.sqrt(p2)
    B = 1.0 + 0.5 * p2
    Cx = 10.0 * x * gb * (2.0 + gb) / (1.0 + x * x * p2)
    sq = np.sqrt(1.0 - x)
    tx = 2.0 - x
    F = (B * (17.0 - 3.0 * x * x / tx**2 - Cx) * sq
         + (12.0 * tx - 7.0 * x * x / tx - 3.0 * x**4 / tx**3)
         * np.log((1.0 + sq) / np.sqrt(x)))
    # A6 negative near endpoint
    return (4.0 * ALPHA_FS / (15.0 * w)) * np.maximum(F, 0.0)


def brems_ee_sigma(eps: np.ndarray, gamma: np.ndarray,
                   regime: str | None = None) -> np.ndarray:
    """e-e dsigma/deps, cm^2/erg."""
    w = np.asarray(eps, dtype=float) / MEC2
    g = np.asarray(gamma, dtype=float)
    w, g = np.broadcast_arrays(w, g)
    if regime is None:
        rel = (g - 1.0) >= _T_SWITCH
    elif regime in ("rel", "nr"):
        rel = np.full(g.shape, regime == "rel")
    else:
        raise ValueError("regime must be None, 'rel' or 'nr'")
    ok = (w > 0) & (g > 1.0) & (w < brems_ee_eps_max(g))
    ok &= rel | (w < 0.25 * (g * g - 1.0))
    ww = np.where(ok, w, 0.25)
    gg = np.where(ok, g, 2.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        val = np.where(rel, _ee_sigma_rel(ww, gg), _ee_sigma_nr(ww, gg))
    return np.where(ok, val * R0**2 / MEC2, 0.0)


# e-p bremsstrahlung
def _atanh_minus_x_over_x3(p: np.ndarray, E: np.ndarray) -> np.ndarray:
    """(atanh x - x) / x^3, x = p/E."""
    x = p / E
    small = x < 1e-2
    xs = np.where(small, x, 0.0)
    x2 = xs * xs
    ser = 1 / 3 + x2 * (1 / 5 + x2 * (1 / 7 + x2 * (1 / 9 + x2 / 11)))
    xl = np.where(small, 0.5, x)
    return np.where(small, ser, (np.arcsinh(p) - xl) / xl**3)


def brems_ep_sigma(eps: np.ndarray, gamma: np.ndarray, Z: float = 1.0,
                   elwert: bool = True) -> np.ndarray:
    """Bethe-Heitler dsigma/deps, cm^2/erg."""
    k = np.asarray(eps, dtype=float) / MEC2
    g = np.asarray(gamma, dtype=float)
    k, g = np.broadcast_arrays(k, g)
    T0 = g - 1.0
    T = T0 - k
    ok = (k > 0) & (T > 0)
    k = np.where(ok, k, 0.5)
    T0 = np.where(ok, T0, 1.0)
    T = np.where(ok, T, 0.5)
    E0 = 1.0 + T0
    E = 1.0 + T
    p0 = np.sqrt(T0 * (T0 + 2.0))
    p = np.sqrt(T * (T + 2.0))
    e0 = 2.0 * np.arcsinh(p0)
    e = 2.0 * np.arcsinh(p)
    L = 2.0 * np.log((T0 + T + T0 * T + p0 * p) / k)
    pp0 = p * p0
    # stable tip combination
    tip = -2.0 / E + 2.0 * _atanh_minus_x_over_x3(p, E) / E**3
    br = (4.0 / 3.0 - 2.0 * E0 * E / p0**2 + E0 * tip + e0 * E / p0**3
          - e * e0 / pp0
          + L * (8.0 * E0 * E / (3.0 * pp0)
                 + k * k * (E0 * E0 * E * E + pp0 * pp0) / pp0**3
                 + k / (2.0 * pp0) * (e0 * (E0 * E + p0 * p0) / p0**3
                                      - e * (E0 * E + p * p) / p**3
                                      + 2.0 * k * E0 * E / pp0**2)))
    val = (Z * Z * ALPHA_FS * R0**2 / MEC2) * (p / p0) * br / k
    if elwert:
        b0, b1 = p0 / E0, p / E
        a = 2.0 * np.pi * Z * ALPHA_FS
        val = val * (b0 / b1) * np.expm1(-a / b0) / np.expm1(-a / b1)
    return np.where(ok, val, 0.0)


# p-p cross sections
_MODELS = ("Geant4", "Pythia8", "SIBYLL", "QGSJET")
_T_TRANS = {"Geant4": np.inf, "Pythia8": 50.0, "SIBYLL": 100.0, "QGSJET": 100.0}
# KATV14 Table IV
_A_MULT = {"Geant4": (0.728, 0.596, 0.491, 0.2503, 0.117),
           "Pythia8": (0.652, 0.0016, 0.488, 0.1928, 0.483),
           "SIBYLL": (5.436, 0.254, 0.072, 0.075, 0.166),
           "QGSJET": (0.908, 0.0009, 6.089, 0.176, 0.448)}
# KATV14 Table VII
_B_AMAX = {"Geant4_0": (9.53, 0.52, 0.054),
           "Geant4": (9.13, 0.35, 9.7e-3),
           "Pythia8": (9.06, 0.3795, 0.01105),
           "SIBYLL": (10.77, 0.412, 0.01264),
           "QGSJET": (13.16, 0.4419, 0.01439)}
# KATV14 Table V
_F_HI = {"Pythia8": (3.5, 0.5, 4.0, 1.0),
         "SIBYLL": (3.55, 0.5, 3.6, 1.0),
         "QGSJET": (3.55, 0.5, 4.5, 1.0)}
_TP_BREAKS = (0.56, 1.0, 2.0, 4.0, 5.0, 20.0, 50.0, 100.0, 1e3)  # GeV


def _check_model(model: str) -> str:
    if model not in _MODELS:
        raise ValueError(f"unknown model {model!r}; use one of {_MODELS}")
    return model


def _sigma_inel_gev(Tp: np.ndarray) -> np.ndarray:
    """KATV14 Eq. 1, mb."""
    Tp = np.asarray(Tp, dtype=float)
    ok = Tp > _TTH
    t = np.where(ok, Tp, 2.0 * _TTH)
    L = np.log(t / _TTH)
    s = (30.7 - 0.96 * L + 0.18 * L * L) * (1.0 - (_TTH / t) ** 1.9) ** 3
    return np.where(ok, s, 0.0)


def pp_inelastic_sigma(Tp: np.ndarray) -> np.ndarray:
    """KATV14 inelastic p-p, cm^2."""
    return _sigma_inel_gev(np.asarray(Tp, dtype=float) / _GEV) * _MB


def _kin(Tp: np.ndarray):
    """KATV14 Eq. 10 kinematics, GeV."""
    s = 2.0 * _MP * (Tp + 2.0 * _MP)
    rs = np.sqrt(s)
    dT = np.maximum(Tp - _TTH, 0.0)
    E_cm = (s - 4.0 * _MP**2 + M_PI0**2) / (2.0 * rs)
    P_cm = np.sqrt(2.0 * _MP * dT * (s - (2.0 * _MP - M_PI0) ** 2)) / (2.0 * rs)
    g_cm = (Tp + 2.0 * _MP) / rs
    gb_cm = np.sqrt(Tp * (Tp + 2.0 * _MP)) / rs
    E_lab = g_cm * E_cm + gb_cm * P_cm
    return s, rs, P_cm, E_lab


def _mult_hi(Tp: np.ndarray, a) -> np.ndarray:
    """KATV14 Eq. 7."""
    xi = np.maximum((Tp - 3.0) / _MP, 1e-300)
    a1, a2, a3, a4, a5 = a
    return (a1 * xi**a4 * (1.0 + np.exp(-a2 * xi**a5))
            * (1.0 - np.exp(-a3 * xi**0.25)))


def _sigma_pi_gev(Tp: np.ndarray, model: str,
                  Tb: np.ndarray | float | None = None,
                  kin=None, sin: np.ndarray | None = None) -> np.ndarray:
    """KATV14 inclusive pi0, mb."""
    Tp = np.asarray(Tp, dtype=float)
    Tb = Tp if Tb is None else np.asarray(Tb, dtype=float)
    _, rs, P_cm, _ = _kin(Tp) if kin is None else kin
    sin = _sigma_inel_gev(Tp) if sin is None else sin

    def low():
        # KATV14 Eqs. 2-5
        eta = P_cm / M_PI0
        Mr, Gr = 1.1883, 0.2264
        gr = np.sqrt(Mr**2 * (Mr**2 + Gr**2))
        Kbw = np.sqrt(8.0) * Mr * Gr * gr / (np.pi * np.sqrt(Mr**2 + gr))
        fbw = _MP * Kbw / (((rs - _MP) ** 2 - Mr**2) ** 2 + Mr**2 * Gr**2)
        s1 = 7.66e-3 * eta**1.95 * (1.0 + eta + eta**5) * fbw**1.86
        s2 = np.where(Tb >= 0.56, 5.7 / (1.0 + np.exp(-9.3 * (Tp - 1.4))), 0.0)
        return s1 + s2

    def mid():
        Qp = (Tp - _TTH) / _MP
        return sin * (-6e-3 + 0.237 * Qp - 0.023 * Qp * Qp)

    def hi(key):
        with np.errstate(over="ignore", invalid="ignore"):
            return sin * _mult_hi(Tp, _A_MULT[key])

    if Tb.ndim == 0:
        tb = float(Tb)
        out = (low() if tb < 2.0 else mid() if tb < 5.0
               else hi("Geant4") if tb < _T_TRANS[model] else hi(model))
    else:
        out = np.where(Tb < 2.0, low(),
                       np.where(Tb < 5.0, mid(),
                                np.where(Tb < _T_TRANS[model], hi("Geant4"),
                                         hi(model))))
    return np.where(Tp > _TTH, out, 0.0)


def pp_pi0_sigma(Tp: np.ndarray, model: str = "Pythia8") -> np.ndarray:
    """KATV14 inclusive pi0 sigma, cm^2."""
    return _sigma_pi_gev(np.asarray(Tp, dtype=float) / _GEV,
                         _check_model(model)) * _MB


def pp_pi0_multiplicity(Tp: np.ndarray, model: str = "Pythia8") -> np.ndarray:
    """Mean pi0 multiplicity."""
    t = np.asarray(Tp, dtype=float) / _GEV
    sin = _sigma_inel_gev(t)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sin > 0, _sigma_pi_gev(t, _check_model(model))
                        / np.where(sin > 0, sin, 1.0), 0.0)


def _amax_gev(Tp: np.ndarray, model: str, Tb, E_lab=None, sp=None) -> np.ndarray:
    """KATV14 Eq. 12, mb/GeV."""
    E_lab = _kin(Tp)[3] if E_lab is None else E_lab
    sp = _sigma_pi_gev(Tp, model, Tb) if sp is None else sp
    th = Tp / _MP
    Tb = np.asarray(Tb, dtype=float)
    if Tb.ndim == 0:
        tb = float(Tb)
        key = (model if tb >= _T_TRANS[model] else "Geant4" if tb >= 5.0
               else "Geant4_0" if tb >= 1.0 else None)
        if key is None:
            return 5.9 * sp / E_lab
        b1, b2, b3 = _B_AMAX[key]
        lt = np.log(th)
        return b1 * th**-b2 * np.exp(b3 * lt * lt) * sp / _MP
    lt = np.log(th)
    out = 5.9 * sp / E_lab
    for key, cond in (("Geant4_0", Tb >= 1.0), ("Geant4", Tb >= 5.0),
                      (model, Tb >= _T_TRANS[model])):
        b1, b2, b3 = _B_AMAX[key]
        out = np.where(cond, b1 * th**-b2 * np.exp(b3 * lt * lt) * sp / _MP, out)
    return out


def pion_gamma_amax(Tp: np.ndarray, model: str = "Pythia8") -> np.ndarray:
    """KATV14 A_max, cm^2/erg."""
    t = np.asarray(Tp, dtype=float) / _GEV
    return _amax_gev(t, _check_model(model), t) * _MB / _GEV


def _f_shape(Tp: np.ndarray, Eg: np.ndarray, model: str, Tb,
             E_lab=None) -> np.ndarray:
    """KATV14 Eqs. 9, 11, 14, 15."""
    E_lab = _kin(Tp)[3] if E_lab is None else E_lab
    X = (Eg - 0.5 * M_PI0) ** 2 / Eg / (E_lab - M_PI0)
    X = np.clip(X, 0.0, 1.0)
    th = Tp / _MP
    kap = 3.29 - 0.2 * th**-1.5

    def mu():
        q = (Tp - 1.0) / _MP
        return 1.25 * np.maximum(q, 0.0) ** 1.25 * np.exp(-1.25 * q)

    Tb = np.asarray(Tb, dtype=float)
    if Tb.ndim == 0:
        tb = float(Tb)
        if model != "Geant4" and tb > _T_TRANS[model]:
            lam, alp, bet, gam = _F_HI[model]
        elif tb > 100.0:
            lam, alp, bet, gam = 3.0, 0.5, 4.9, 1.0
        elif tb > 20.0:
            lam, alp, bet, gam = 3.0, 0.5, 4.2, 1.0
        elif tb > 4.0:
            m = mu()
            lam, alp, bet, gam = 3.0, 1.0, 1.5 * m + 4.95, m + 1.5
        elif tb > 1.0:
            m = mu()
            lam, alp, bet, gam = 3.0, 1.0, m + 2.45, m + 1.45
        else:
            lam, alp, bet, gam = 1.0, 1.0, kap, 0.0
    else:
        m = mu()
        one = np.ones_like(Tp)
        lam, alp, bet, gam = 1.0 * one, 1.0 * one, kap, 0.0 * one
        table = [(Tb > 1.0, (3.0, 1.0, m + 2.45, m + 1.45)),
                 (Tb > 4.0, (3.0, 1.0, 1.5 * m + 4.95, m + 1.5)),
                 (Tb > 20.0, (3.0, 0.5, 4.2, 1.0)),
                 (Tb > 100.0, (3.0, 0.5, 4.9, 1.0))]
        if model != "Geant4":
            table.append((Tb > _T_TRANS[model], _F_HI[model]))
        for cond, (l_, a_, b_, g_) in table:
            lam = np.where(cond, l_, lam)
            alp = np.where(cond, a_, alp)
            bet = np.where(cond, b_, bet)
            gam = np.where(cond, g_, gam)
    Cc = lam * M_PI0 / E_lab
    return (1.0 - X**alp) ** bet / (1.0 + X / Cc) ** gam


_SIN_1GEV: float = float(_sigma_inel_gev(1.0))
_SIN_1TEV: float = float(_sigma_inel_gev(1e3))


def _nuclear_gev(Tp: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """KATV14 Eq. 24, Tp in GeV."""
    sin = np.where(Tp >= 1.0, sin, _SIN_1GEV)
    G = 1.0 + np.log(np.maximum(1.0, sin / _SIN_1TEV))
    return 1.37 + (0.29 + 0.1) * (10.0 * np.pi) * G / sin


def pp_nuclear_enhancement(Tp: np.ndarray) -> np.ndarray:
    """KATV14 Eq. 24, ISM abundances."""
    t = np.maximum(np.asarray(Tp, dtype=float) / _GEV, 1.0)
    return _nuclear_gev(t, _sigma_inel_gev(t))


def pion_gamma_emax(Tp: np.ndarray) -> np.ndarray:
    """KATV14 Eq. 10 photon endpoint, erg."""
    t = np.asarray(Tp, dtype=float) / _GEV
    _, _, _, E_lab = _kin(np.maximum(t, _TTH))
    gpi = E_lab / M_PI0
    out = 0.5 * M_PI0 * (gpi + np.sqrt(gpi * gpi - 1.0)) * _GEV
    return np.where(t >= _TTH, out, 0.0)


def _tp_min_gev(Eg: np.ndarray) -> np.ndarray:
    """Tp with E_pi,max = Y_gamma."""
    Eg = np.asarray(Eg, dtype=float)
    Y = Eg + M_PI0**2 / (4.0 * Eg)
    above = Y > _kin(np.array(_TTH))[3]
    lo = np.full(Eg.shape, np.log(1e-12))
    hi = np.log(np.maximum(4.0 * Y, 10.0))
    for _ in range(80):
        m = 0.5 * (lo + hi)
        f = _kin(_TTH + np.exp(m))[3] - Y
        lo = np.where(f < 0, m, lo)
        hi = np.where(f < 0, hi, m)
    return np.where(above, _TTH + np.exp(0.5 * (lo + hi)), _TTH)


def pion_tp_min(eps: np.ndarray) -> np.ndarray:
    """Minimum Tp for photon energy, erg."""
    return _tp_min_gev(np.asarray(eps, dtype=float) / _GEV) * _GEV


def _dsigma_gev(Eg, Tp, model, Tb, nuclear) -> np.ndarray:
    """dsigma/dEg, mb/GeV."""
    kin = _kin(Tp)
    sin = _sigma_inel_gev(Tp)
    sp = _sigma_pi_gev(Tp, model, Tb, kin, sin)
    E_lab = kin[3]
    val = _amax_gev(Tp, model, Tb, E_lab, sp) * _f_shape(Tp, Eg, model, Tb, E_lab)
    if nuclear is True:
        val = val * _nuclear_gev(Tp, sin)
    elif nuclear is not False:
        val = val * float(nuclear)
    return val


def pion_gamma_dsigma(eps: np.ndarray, Tp: np.ndarray, model: str = "Pythia8",
                      nuclear_enhancement=False) -> np.ndarray:
    """KATV14 dsigma/dE_gamma, cm^2/erg."""
    model = _check_model(model)
    Eg = np.asarray(eps, dtype=float) / _GEV
    t = np.asarray(Tp, dtype=float) / _GEV
    Eg, t = np.broadcast_arrays(Eg, t)
    ok = (t > _TTH) & (Eg > 0)
    tt = np.where(ok, t, 2.0)
    ee = np.where(ok, Eg, 0.1)
    ok &= (ee - 0.5 * M_PI0) ** 2 / ee < _kin(tt)[3] - M_PI0
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        val = _dsigma_gev(ee, tt, model, tt, nuclear_enhancement)
    return np.where(ok, val * _MB / _GEV, 0.0)
