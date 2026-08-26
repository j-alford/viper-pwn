"""Deprecated alias for :mod:`viper.scalar_params`."""

import warnings

from .scalar_params import PWNScalarParams, check_physical_inputs  # noqa: F401

warnings.warn(
    "viper.legacy_params was renamed to viper.scalar_params "
    "(PWNModelParams is now PWNScalarParams); this alias module "
    "will be removed at the first public release.",
    DeprecationWarning,
    stacklevel=2,
)

PWNModelParams = PWNScalarParams
physics_problems = check_physical_inputs
