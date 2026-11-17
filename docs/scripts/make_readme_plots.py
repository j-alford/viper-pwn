"""Render the README example plot.

Generates one PNG under docs/images/:
  sed_example.png        SED + evolved particle spectrum on dual x-axes

Uses the README Quick Start model.
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import astropy.units as u

from viper import PWNModel
from viper.plotting import plot_model_sed

# same text sizes as make_evolution_grid.py
matplotlib.rcParams.update({'font.size': 14, 'axes.labelsize': 16,
                            'xtick.labelsize': 14, 'ytick.labelsize': 14,
                            'legend.fontsize': 14})

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, '..', 'images'))
os.makedirs(OUT, exist_ok=True)

# the README Quick Start model
model = PWNModel()
model.set_pulsar('J0007+7303')
model.distance.value = 1.4 * u.kpc
model.esn.value      = 1e51 * u.erg
model.mej.value      = 6 * u.M_sun
model.n_ism.value    = 0.4 / u.cm**3
model.tau_sd.value   = 3000 * u.yr
model.eta_b.value    = 0.01
model.e_max.value    = 500 * u.TeV

result = model.run(dyn_step=10)

# --- SED + particles ---
fig, axes = plot_model_sed(result, distance=model.distance.value,
                           model_label='SED')
axes['sed'].get_legend().set_loc('lower center')
fig.subplots_adjust(left=0.105, right=0.905, top=0.87, bottom=0.115)
fig.savefig(os.path.join(OUT, 'sed_example.png'), dpi=120)
plt.close(fig)

print(f"Wrote {OUT}")
