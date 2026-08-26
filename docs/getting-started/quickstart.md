# Quick start

Set up a model, run it, and read off the final state:

```python
import astropy.units as u
from viper import PWNModel

model = PWNModel()
model.set_pulsar('J0007+7303')           # loads P, Pdot, distance from ATNF catalog
model.distance.value = 1.4 * u.kpc       # override the ATNF catalog value if desired

model.esn.value = 1e51 * u.erg
model.mej.value = 6.0 * u.M_sun
model.n_ism.value = 0.4 / u.cm**3
model.braking_index.value = 2.5
model.tau_sd.value = 3000 * u.yr
model.eta_b.value = 0.01
model.e_min.value = 3.0 * u.GeV
model.e_max.value = 500 * u.TeV
model.e_break.value = 50.0 * u.GeV
model.p1.value = 0.8
model.p2.value = 2.5

result = model.run()

dyn = result.dyninfo[-1]
print(f"R_pwn = {dyn.rad_pwn:.2f} pc")
print(f"B     = {dyn.bpwn:.2f} uG")
print(f"V_pwn = {dyn.vpwn:.0f} km/s")
```

Additional IC seed photon fields register with
`model.add_ic_field(temp=..., norm=...)`, where `norm` is the field
energy density in eV/cm^3. The CMB is always included.

## Plotting the SED

```python
import matplotlib.pyplot as plt
from viper.plotting import plot_model_sed

fig, _ = plot_model_sed(result, distance=model.distance.value)
plt.show()
```

![Broadband SED example](../images/sed_example.png)

## Evolution history

`result.dyninfo` records the full dynamical history at every
timestep: radii (`rad_pwn`, `rad_snr`, `rad_rs`, `rad_psr`),
velocities (`vpwn`, `vejpwn`, `vsnr`), magnetic field (`bpwn`),
energies (`epwn_50`, `epwnb_50`, `epwnp_50`, in units of 10^50 erg),
luminosities (`synch_lum_40`, `ic_lum_40`, `ad_lum_40`, `edot_40`, in
10^40 erg/s), pressures (`ppwn`, `psnrpwn`), and the PWN-swept ejecta
mass (`mswpwn`).

![PWN evolution history grid](../images/evolution_grid.png)
