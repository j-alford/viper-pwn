"""Render the README example plot.

Generates one PNG under docs/images/:
  sed_example.png        SED + evolved particle spectrum on dual x-axes

Parameters are G54.1-like round-number demo values, not a published fit.
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import astropy.units as u

from viper import PWNModel
from viper.plotting import plot_model_sed

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, '..', 'images'))
os.makedirs(OUT, exist_ok=True)

model = PWNModel()
model.set_pulsar('J1930+1852')
model.distance.value      = 6.0 * u.kpc
model.esn.value           = 1e51 * u.erg
model.mej.value           = 20.0 * u.M_sun
model.n_ism.value         = 0.01 / u.cm**3
model.braking_index.value = 2.0
model.tau_sd.value        = 3500 * u.yr
model.eta_b.value         = 3e-3
model.e_min.value         = 10.0 * u.GeV
model.e_max.value         = 500 * u.TeV
model.e_break.value       = 3.0 * u.TeV
model.p1.value            = 2.3
model.p2.value            = 2.4
model.add_ic_field(temp=2750, norm=27.0)   # NIR field [eV/cm^3]

result = model.run(dyn_step=10)

# --- SED + particles ---
fig, _ = plot_model_sed(result, distance=model.distance.value,
                        model_label='SED')
fig.savefig(os.path.join(OUT, 'sed_example.png'), dpi=120,
            bbox_inches='tight')
plt.close(fig)

print(f"Wrote {OUT}")
