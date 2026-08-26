"""
Physical parameters require astropy Quantities; dimensionless ones
accept bare floats.  Log-space transforms for the MCMC sampler are
handled via ``fit_value`` and ``fit_bounds``.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class Parameter:
    """A single model parameter with optional unit enforcement.

    Parameters
    ----------
    name : str
        Parameter name (must match the model's internal convention).
    default : float
        Default value in internal units; bypasses unit validation.
    frozen : bool
        If True (default), the parameter is held fixed during fitting.
    log : bool
        If True, the MCMC sampler works in log10 space.
    bounds : tuple of float, optional
        (lo, hi) in internal physical units; converted to fit space
        by ``fit_bounds``.
    prior : tuple of float, optional
        Gaussian prior ``(mu, sigma)``, applied in physical space by
        default; set ``prior_log=True`` for fit (log10) space.  None
        means flat within bounds.
    unit : astropy unit, optional
        If set, ``value`` requires an astropy Quantity converted to
        this unit; if None, bare floats are accepted.
    scale : float
        Multiplicative factor after unit conversion:
        ``internal = quantity.to(unit).value * scale``.
    description : str
        Human-readable description for display.
    """

    __slots__ = (
        'name', '_value', 'frozen', 'log', '_bounds', '_prior', '_prior_log',
        '_unit', '_scale', 'description',
    )

    def __init__(
        self,
        name: str,
        default: float = 0.0,
        frozen: bool = True,
        log: bool = False,
        bounds: Optional[tuple[float, float]] = None,
        prior: Optional[tuple[float, float]] = None,
        unit=None,
        scale: float = 1.0,
        description: str = '',
    ) -> None:
        self.name = name
        self._value = float(default)
        self.frozen = frozen
        self.log = log
        self._prior_log = False
        self._unit = unit
        self._scale = scale
        self.description = description
        # Constructor bounds: internal floats, validated.
        self._bounds = None
        if bounds is not None:
            lo, hi = bounds
            self._bounds = self._checked_bounds(float(lo), float(hi))
        self._prior = None
        self.prior = prior

    # Value access

    @property
    def value(self) -> float:
        """Current value in internal physical units."""
        return self._value

    @value.setter
    def value(self, val) -> None:
        """Set value.  Physical parameters require astropy Quantities."""
        if self._unit is not None:
            self._value = self._convert_quantity(val, context='value')
        else:
            if _is_quantity(val):
                raise ValueError(
                    f"Parameter '{self.name}' is dimensionless. "
                    f"Pass a bare number, not a Quantity ({val})."
                )
            self._value = float(val)

    # Bounds access

    @property
    def bounds(self) -> Optional[tuple[float, float]]:
        """Bounds in internal physical units, or None."""
        return self._bounds

    @bounds.setter
    def bounds(self, val) -> None:
        """Set bounds.  Physical parameters require astropy Quantities."""
        if val is None:
            self._bounds = None
            return
        lo, hi = val
        if self._unit is not None:
            lo_f = self._convert_quantity(lo, context='lower bound')
            hi_f = self._convert_quantity(hi, context='upper bound')
        else:
            if _is_quantity(lo) or _is_quantity(hi):
                raise ValueError(
                    f"Parameter '{self.name}' is dimensionless. "
                    f"Pass bare numbers for bounds."
                )
            lo_f, hi_f = float(lo), float(hi)
        self._bounds = self._checked_bounds(lo_f, hi_f)

    def _checked_bounds(self, lo_f: float, hi_f: float) -> tuple[float, float]:
        """Sanity-check bounds already converted to internal floats.

        NaN compares False, hence the explicit finiteness check.
        """
        if not (np.isfinite(lo_f) and np.isfinite(hi_f)):
            raise ValueError(
                f"Bounds for '{self.name}' must be finite, "
                f"got ({lo_f}, {hi_f})."
            )
        if lo_f >= hi_f:
            raise ValueError(
                f"Lower bound ({lo_f}) must be < upper bound ({hi_f}) "
                f"for '{self.name}'."
            )
        if self.log and lo_f <= 0:
            raise ValueError(
                f"Parameter '{self.name}' is log-sampled; bounds must be "
                f"positive, got ({lo_f}, {hi_f})."
            )
        return (lo_f, hi_f)

    # Prior access

    @property
    def prior(self) -> Optional[tuple[float, float]]:
        """Gaussian prior (mu, sigma), or None.

        Physical space by default; fit (log10) space when
        ``prior_log`` is True.
        """
        return self._prior

    @prior.setter
    def prior(self, val) -> None:
        if val is None:
            self._prior = None
            return
        mu, sigma = val
        mu, sigma = float(mu), float(sigma)
        if not (np.isfinite(mu) and np.isfinite(sigma)) or sigma <= 0:
            raise ValueError(
                f"Prior for '{self.name}' must have finite mu and "
                f"sigma > 0, got (mu={mu}, sigma={sigma})."
            )
        self._prior = (mu, sigma)

    @property
    def prior_log(self) -> bool:
        """If True, prior is applied in log10 space (on theta directly)."""
        return self._prior_log

    @prior_log.setter
    def prior_log(self, val: bool) -> None:
        self._prior_log = val

    # Fit-space interface

    @property
    def fit_value(self) -> float:
        """Value in MCMC sampling space (log10 if ``log=True``)."""
        if self.log:
            if self._value <= 0:
                return -np.inf
            return np.log10(self._value)
        return self._value

    @property
    def fit_bounds(self) -> tuple[float, float]:
        """Bounds in MCMC sampling space."""
        if self._bounds is None:
            return (-np.inf, np.inf)
        lo, hi = self._bounds
        if self.log:
            # log may flip after bounds set.
            if lo <= 0:
                raise ValueError(
                    f"Parameter '{self.name}' is log-sampled but has "
                    f"nonpositive lower bound {lo}; set positive bounds "
                    f"or log=False."
                )
            return (np.log10(lo), np.log10(hi))
        return (lo, hi)

    def set_from_fit_value(self, fit_val: float) -> None:
        """Update internal value from an MCMC sample (fit space)."""
        if self.log:
            self._value = 10.0 ** fit_val
        else:
            self._value = fit_val

    # Internal: unit conversion

    def _convert_quantity(self, val, context: str = 'value') -> float:
        """Convert an astropy Quantity to internal float."""
        import astropy.units as u
        if not isinstance(val, u.Quantity):
            raise ValueError(
                f"Parameter '{self.name}' requires an astropy Quantity "
                f"with units compatible with {self._unit} for {context}. "
                f"Got bare {type(val).__name__}: {val}. "
                f"Example: {val} * u.{_unit_hint(self._unit)}"
            )
        try:
            converted = val.to(self._unit).value
        except u.UnitConversionError as e:
            raise ValueError(
                f"Cannot convert {val.unit} to {self._unit} "
                f"for parameter '{self.name}': {e}"
            ) from None
        return float(converted * self._scale)

    def _set_internal(self, value: float) -> None:
        """Bypass validation and set internal value directly."""
        self._value = float(value)

    # Display

    @property
    def fit_label(self) -> str:
        """Label for MCMC plots: 'log(name)' if log, else 'name'."""
        return f'log({self.name})' if self.log else self.name

    def __repr__(self) -> str:
        parts = [f"Parameter('{self.name}', value={self._value}"]
        if self.frozen:
            parts.append(', frozen')
        else:
            parts.append(', free')
        if self.log:
            parts.append(', log')
        if self._bounds is not None:
            parts.append(f', bounds={self._bounds}')
        if self._unit is not None:
            parts.append(f', unit={self._unit}')
            if self._scale != 1.0:
                parts.append(f', scale={self._scale}')
        parts.append(')')
        return ''.join(parts)


class ParameterSet:
    """Ordered collection of Parameters with fit-space helpers.

    Provides dict-like access by name and array-based interfaces for
    the MCMC sampler.
    """

    def __init__(self, params: list[Parameter]) -> None:
        self._params: list[Parameter] = list(params)
        self._index: dict[str, int] = {
            p.name: i for i, p in enumerate(params)
        }

    def __getitem__(self, name: str) -> Parameter:
        try:
            return self._params[self._index[name]]
        except KeyError:
            raise KeyError(
                f"No parameter named '{name}'. "
                f"Available: {list(self._index.keys())}"
            ) from None

    def __contains__(self, name: str) -> bool:
        return name in self._index

    def __iter__(self):
        return iter(self._params)

    def __len__(self) -> int:
        return len(self._params)

    def add(self, param: Parameter) -> None:
        """Add a new parameter to the set."""
        if param.name in self._index:
            raise ValueError(f"Parameter '{param.name}' already exists.")
        self._index[param.name] = len(self._params)
        self._params.append(param)

    def remove(self, name: str) -> None:
        """Remove a parameter by name."""
        if name not in self._index:
            raise KeyError(f"No parameter named '{name}'.")
        idx = self._index[name]
        del self._params[idx]
        self._index = {p.name: i for i, p in enumerate(self._params)}

    @property
    def names(self) -> list[str]:
        """All parameter names in order."""
        return [p.name for p in self._params]

    # Free-parameter interface

    @property
    def free(self) -> list[Parameter]:
        """List of free (unfrozen) parameters."""
        return [p for p in self._params if not p.frozen]

    @property
    def ndim(self) -> int:
        """Number of free parameters."""
        return len(self.free)

    @property
    def free_names(self) -> list[str]:
        """Names of free parameters in order."""
        return [p.name for p in self.free]

    @property
    def free_labels(self) -> list[str]:
        """Plot labels for free parameters."""
        return [p.fit_label for p in self.free]

    @property
    def free_fit_values(self) -> np.ndarray:
        """Fit-space values for all free parameters."""
        return np.array([p.fit_value for p in self.free])

    @property
    def free_fit_bounds(self) -> list[tuple[float, float]]:
        """Fit-space bounds for all free parameters."""
        return [p.fit_bounds for p in self.free]

    def update_free_from_theta(self, theta: np.ndarray) -> None:
        """Update free parameters from an MCMC theta vector."""
        free = self.free
        if len(theta) != len(free):
            raise ValueError(
                f"theta has {len(theta)} elements but there are "
                f"{len(free)} free parameters."
            )
        for i, p in enumerate(free):
            p.set_from_fit_value(theta[i])

    def theta_to_physical(self, theta: np.ndarray) -> dict[str, float]:
        """Convert a theta vector to a dict of physical values."""
        free = self.free
        result = {}
        for i, p in enumerate(free):
            val = theta[i]
            if p.log:
                result[p.name] = 10.0 ** val
            else:
                result[p.name] = val
        return result

    # Display

    def __repr__(self) -> str:
        lines = []
        hdr = (f"{'Name':<18s} {'Value':>12s} {'Status':>8s} "
               f"{'Log':>5s} {'Bounds':>24s}")
        lines.append(hdr)
        lines.append('-' * len(hdr))
        for p in self._params:
            status = 'frozen' if p.frozen else 'free'
            log_str = 'yes' if p.log else ''
            if p._bounds is not None:
                lo, hi = p._bounds
                if p.log:
                    bnd = f'({lo:.3g}, {hi:.3g})'
                else:
                    bnd = f'({lo:.4g}, {hi:.4g})'
            else:
                bnd = ''
            lines.append(
                f"{p.name:<18s} {p._value:>12.4g} {status:>8s} "
                f"{log_str:>5s} {bnd:>24s}"
            )
        return '\n'.join(lines)


# Utility

def _is_quantity(val) -> bool:
    """Check if val is an astropy Quantity without importing astropy."""
    return hasattr(val, 'to') and hasattr(val, 'unit')


def _unit_hint(unit) -> str:
    """Generate a short hint string for an astropy unit."""
    s = str(unit)
    if s == 'erg':
        return 'erg'
    if s == 'pc':
        return 'pc'
    if s == 'GeV':
        return 'GeV'
    return s
