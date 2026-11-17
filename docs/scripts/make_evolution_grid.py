"""Render the README "Evolution history" four-panel figure.

Writes docs/images/evolution_grid.png.
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
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
v_snr    = np.array([d.vsnr     for d in dyn])
bpwn     = np.array([d.bpwn     for d in dyn])
e_b_50   = np.array([d.epwnb_50 for d in dyn])
e_p_50   = np.array([d.epwnp_50 for d in dyn])

t_pos = _pos(t)
T_MAX = float(np.nanmax(t_pos))


mpl.rcParams['font.size'] = 14
GREY = '0.55'
LEG  = dict(fontsize=14, frameon=False, ncol=3, handlelength=2.2,
            columnspacing=1.6)

# intervals with the nebula shrinking
neg = np.concatenate(([False], v_pwn < 0, [False]))
edges = np.flatnonzero(neg[1:] != neg[:-1]).reshape(-1, 2)
shrink = [(t[i], t[j - 1]) for i, j in edges]

fig, axes = plt.subplots(4, 1, figsize=(10, 8.5), sharex=True, dpi=120,
                         gridspec_kw=dict(hspace=0))
for ax in axes:
    ax.set_xscale('log')
    ax.set_xlim(10, T_MAX)
    for t0, t1 in shrink:
        ax.axvspan(t0, t1, color='0.93', lw=0, zorder=0)
    ax.grid(which='major', color='0.9', lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(which='both', direction='in', top=True, right=True,
                   labelsize=14)
    ax.tick_params(which='major', length=5)
    ax.tick_params(which='minor', length=2.5)
axes[-1].set_xlabel('Time [yr]', fontsize=16)

# --- radii ---
ax = axes[0]
ax.plot(t_pos, _pos(rad_snr), color=RED,  ls='-',  lw=1.6, label=r'$R_\mathrm{SNR}$')
ax.plot(t_pos, _pos(rad_rs),  color=RED,  ls='--', lw=1.6, label=r'$R_\mathrm{RS}$')
ax.plot(t_pos, _pos(rad_pwn), color=BLUE, ls='-',  lw=1.6, label=r'$R_\mathrm{PWN}$')
ax.set_yscale('log')
ax.set_ylim(4e-4, 3e2)
ax.set_ylabel('Radius [pc]', fontsize=16)
if shrink:
    ax.text(shrink[0][0] * 0.93, 1.2e-2, 'compression by\nreverse shock',
            ha='right', va='center', fontsize=13, color='0.35')
ax.legend(loc='upper left', **LEG)

# --- velocities ---
ax = axes[1]
ax.axhline(0, color=GREY, lw=0.6)
ax.plot(t_pos, v_snr,   color=RED,  ls='-',  lw=1.6, label=r'$v_\mathrm{SNR}$')
ax.plot(t_pos, v_pwn,   color=BLUE, ls='-',  lw=1.6, label=r'$v_\mathrm{PWN}$')
ax.set_yscale('symlog', linthresh=100, linscale=1.0)
ax.set_yticks([-1e3, -1e2, 0, 1e2, 1e3, 1e4])
ax.set_ylim(-6e3, 1e5)
ax.set_ylabel(r'Velocity [km s$^{-1}$]', fontsize=16)
ax.legend(loc='lower left', **LEG)

# --- B-field ---
ax = axes[2]
ax.plot(t_pos, _pos(bpwn), color=BLUE, ls='-', lw=1.6, label=r'$B_\mathrm{PWN}$')
ax.set_yscale('log')
ax.set_ylim(1, 1e6)
ax.set_yticks([1e1, 1e3, 1e5])
ax.set_ylabel(r'$B$ [$\mu$G]', fontsize=16)
ax.legend(loc='lower left', **LEG)

# --- energies (10^50 erg -> erg) ---
ax = axes[3]
ax.plot(t_pos, _pos(e_p_50) * 1e50, color=BLUE, ls='-',  lw=1.6, label=r'$E_\mathrm{particles}$')
ax.plot(t_pos, _pos(e_b_50) * 1e50, color=BLUE, ls='--', lw=1.6, label=r'$E_B$')
ax.set_yscale('log')
ax.set_ylim(1e43, 1e50)
ax.set_yticks([1e44, 1e46, 1e48])
ax.set_ylabel('Energy [erg]', fontsize=16)
ax.legend(loc='upper left', **LEG)

for ax in (axes[0], axes[2], axes[3]):
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(1.0,), numticks=20))
    ax.yaxis.set_minor_formatter(NullFormatter())

fig.align_ylabels(axes)
fig.subplots_adjust(left=0.105, right=0.985, top=0.985, bottom=0.075)

out_path = os.path.join(OUT, 'evolution_grid.png')
fig.savefig(out_path, dpi=120)
plt.close(fig)
print(f"Wrote {out_path}")
