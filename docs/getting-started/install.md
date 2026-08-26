# Installation

VIPER requires Python >= 3.10, NumPy >= 1.21, SciPy >= 1.7, and
Astropy >= 5.0.

## From PyPI

```bash
pip install viper-pwn
```

Optional extras:

| Extra     | Installs      | Enables                                  |
|-----------|---------------|------------------------------------------|
| `fast`    | numba         | JIT-compiled radiation kernels           |
| `plot`    | matplotlib    | SED and evolution plotting               |
| `fit`     | emcee, corner | MCMC fitting                             |
| `gammapy` | gammapy       | the `PWNSpectralModel` adapter           |
| `nested`  | ultranest     | nested-sampling fits                     |
| `all`     | all of the above |                                       |

```bash
pip install "viper-pwn[fast,plot,fit]"
```

Numba is optional but recommended: a full evolution of a typical,
middle-aged PWN (e.g. CTA 1) takes ~4 s in pure NumPy and ~< 1 s
with numba.

## Development install

```bash
git clone https://github.com/j-alford/viper-pwn
cd viper-pwn
uv venv && source .venv/bin/activate
uv pip install -e ".[all]"
```
