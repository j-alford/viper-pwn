"""
SED plotting and parameter display utilities for PWN model fitting.

Usage
-----
    from viper.plotting import plot_sed, plot_electron_spectrum, compute_luminosities

    # Quick SED plot at a given theta
    plot_sed(config, theta)

    # With a data-file overlay (see load_sed_data for the column format)
    plot_sed(config, theta, data_file='my_source_sed.txt')
"""

from . import (
    style,
    data,
    sed,
    particle_sed,
    movie,
    analysis,
    electrons,
    corner,
)
from .data import load_sed_data
from .sed import plot_sed, plot_model_sed
from .particle_sed import plot_sed_with_particles
from .movie import plot_model_sed_movie
from .analysis import compute_luminosities
from .electrons import plot_electron_spectrum
from .corner import plot_corner

__all__ = [
    'load_sed_data',
    'plot_sed',
    'plot_model_sed',
    'plot_sed_with_particles',
    'plot_model_sed_movie',
    'compute_luminosities',
    'plot_electron_spectrum',
    'plot_corner',
]

# Back-compat: re-export every submodule name; first definition wins.
for _m in (style, data, sed, particle_sed, movie, analysis, electrons,
           corner):
    for _k, _v in vars(_m).items():
        if not _k.startswith('__'):
            globals().setdefault(_k, _v)
del _m, _k, _v
