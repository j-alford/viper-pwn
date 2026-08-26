# VIPER

**Volume-Integrated PWN Evolution & Radiation**: a Python
implementation of the Gelfand, Slane & Zhang (2009) pulsar wind nebula
model. VIPER evolves a pulsar wind nebula inside its supernova
remnant, tracking the PWN radius, magnetic field, particle energy
spectrum, and broadband photon emission from radio through gamma rays.

![Broadband SED example](images/sed_example.png)

Features:

- VIPER is a full, self-consistent evolutionary model
  - SNR dynamics
  - PWN interaction with the reverse shock
  - track history of physical quantities (magnetic field, shock speed,
    pressure etc.)
  - volume-integrated particle spectrum.
- Calculate broadband SED with
  - synchrotron
  - inverse Compton on the CMB + (optional) photon fields
  - Synchrotron Self Compton (optional)
  - particle escape (optional)
- Fitting:
  - unit-aware parameters with bounds and priors
  - automatic unit conversion of observable quantities
  - MCMC (emcee)
  - nested sampling (ultranest)
- Speed:
  - numba-compiled kernels run a full evolution in < 1s on an M series
    MacBook Pro.
  - multi-threading support

```{toctree}
:hidden:

getting-started/index
tutorials/index
api/index
citing
```
