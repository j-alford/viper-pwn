"""Render the README "Evolution history" 2x2 grid figure.

Writes docs/images/evolution_grid.png.
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u

from viper import PWNModel


mpl.rcParams['font.family']      = 'serif'
mpl.rcParams['font.serif']       = ['CMU Serif', 'Computer Modern Roman',
                                    'DejaVu Serif']
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['text.usetex']      = False

BLUE = 'xkcd:blue'
RED  = 'xkcd:red'

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.abspath(os.path.join(HERE, '..', 'images'))
os.makedirs(OUT, exist_ok=True)


def _pos(arr):
    """Replace non-positive entries with NaN (safe for log axes)."""
    a = np.asarray(arr, dtype=float)
    return np.where(a > 0, a, np.nan)


model = PWNModel()
model.set_pulsar('J0007+7303')
model.distance.value      = 1.4 * u.kpc
model.esn.value           = 1e51 * u.erg
model.mej.value           = 6.0 * u.M_sun
model.n_ism.value         = 0.4 / u.cm**3
model.braking_index.value = 2.5
model.tau_sd.value        = 3000 * u.yr
model.eta_b.value         = 0.01
model.e_min.value         = 3.0 * u.GeV
model.e_max.value         = 500 * u.TeV
model.e_break.value       = 50.0 * u.GeV
model.p1.value            = 0.8
model.p2.value            = 2.5

result = model.run()
dyn    = result.dyninfo

t        = np.array([d.time     for d in dyn])
rad_pwn  = np.array([d.rad_pwn  for d in dyn])
rad_snr  = np.array([d.rad_snr  for d in dyn])
rad_rs   = np.array([d.rad_rs   for d in dyn])
v_pwn    = np.array([d.vpwn     for d in dyn])
v_ejpwn  = np.array([d.vejpwn   for d in dyn])
v_snr    = np.array([d.vsnr     for d in dyn])
bpwn     = np.array([d.bpwn     for d in dyn])
e_pwn_50 = np.array([d.epwn_50  for d in dyn])
e_b_50   = np.array([d.epwnb_50 for d in dyn])
e_p_50   = np.array([d.epwnp_50 for d in dyn])

t_pos = _pos(t)
T_MAX = float(np.nanmax(t_pos))


fig = plt.figure(figsize=(11, 9), constrained_layout=True, dpi=110)
gs  = fig.add_gridspec(2, 2)

# --- top-left: radii ---
ax = fig.add_subplot(gs[0, 0])
ax.loglog(t_pos, _pos(rad_pwn), color=BLUE, ls='-',
          label=r'$R_\mathrm{PWN}$')
ax.loglog(t_pos, _pos(rad_snr), color=RED, ls='-',
          label=r'$R_\mathrm{SNR}$')
ax.loglog(t_pos, _pos(rad_rs),  color=RED, ls='--',
          label=r'$R_\mathrm{RS}$')
ax.set_xlim(1, T_MAX)
ax.set_xlabel('Time [yr]')
ax.set_ylabel('Radius [pc]')
ax.grid(which='major', alpha=0.3)
ax.legend(fontsize=12, frameon=False)

# --- top-right: velocities ---
ax = fig.add_subplot(gs[0, 1])
ax.plot(t_pos, v_pwn,   color=BLUE, ls='-',  label=r'$v_\mathrm{PWN}$')
ax.plot(t_pos, v_ejpwn, color=BLUE, ls='--', label=r'$v_\mathrm{ej,PWN}$')
ax.plot(t_pos, v_snr,   color=RED,  ls='-',  label=r'$v_\mathrm{SNR}$')
ax.set_xscale('log')
ax.set_yscale('symlog', linthresh=1)
ax.set_xlim(1, T_MAX)
ax.set_xlabel('Time [yr]')
ax.set_ylabel('Velocity [km s$^{-1}$]')
ax.grid(which='major', alpha=0.3)
ax.legend(fontsize=12, frameon=False)

# --- bottom-left: B-field ---
ax = fig.add_subplot(gs[1, 0])
ax.loglog(t_pos, _pos(bpwn), color=BLUE, ls='-', label=r'$B_\mathrm{PWN}$')
ax.set_xlim(1, T_MAX)
ax.set_ylim(1, 1e6)
ax.set_xlabel('Time [yr]')
ax.set_ylabel(r'$B$ [$\mu$G]')
ax.grid(which='major', alpha=0.3)
ax.legend(fontsize=12, frameon=False)

# --- bottom-right: energies (10^50 erg -> erg) ---
ax = fig.add_subplot(gs[1, 1])
ax.loglog(t_pos, _pos(e_pwn_50) * 1e50, color=BLUE, ls='-',
          label=r'$E_\mathrm{PWN}$')
ax.loglog(t_pos, _pos(e_b_50)   * 1e50, color=RED,  ls='-',
          label=r'$E_B$')
ax.loglog(t_pos, _pos(e_p_50)   * 1e50, color=RED,  ls='--',
          label=r'$E_p$')
ax.set_xlim(1, T_MAX)
ax.set_xlabel('Time [yr]')
ax.set_ylabel('Energy [erg]')
ax.grid(which='major', alpha=0.3)
ax.legend(fontsize=12, frameon=False)

out_path = os.path.join(OUT, 'evolution_grid.png')
fig.savefig(out_path, dpi=110)
plt.close(fig)
print(f"Wrote {out_path}")
