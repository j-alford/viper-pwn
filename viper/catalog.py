"""
Parses the ATNF Pulsar Catalogue psrcat.db file.

Provides lookup by pulsar J-name (e.g. 'J0007+7303') or B-name (e.g. 'B0531+21').

When a pulsar has F0/F1 (spin frequency and derivative) instead of P0/P1 (period and derivative), conversion is automatic:

    P0 = 1 / F0
    P1 = -F1 / F0**2

Reference: Manchester, R.N., Hobbs, G.B., Teoh, A. & Hobbs, M. (2005),
AJ, 129, 1993.  https://www.atnf.csiro.au/research/pulsar/psrcat/
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
_DEFAULT_DB = os.path.join(_DATA_DIR, 'psrcat.db')


class PulsarEntry:
    """A single pulsar from the ATNF catalogue.

    Attributes
    ----------
    jname : str
        J2000 name (e.g. 'J0007+7303').
    bname : str or None
        B1950 name if available.
    dm : float or None
        Dispersion measure [pc/cm^3].
    dist : float or None
        Best available distance estimate [kpc].
    assoc : str or None
        Association string (e.g. 'SNR:CTA1[hgc+04]').
    raj : str or None
        Right ascension (J2000) string.
    decj : str or None
        Declination (J2000) string.
    raw : dict
        All raw parameter key-value pairs from the catalogue.
    """

    def __init__(self, raw: dict) -> None:
        self.raw = raw
        self.jname: str = raw.get('PSRJ', '')
        self.bname: Optional[str] = raw.get('PSRB', None)

        self._p0 = _float_or_none(raw.get('P0'))
        self._p1 = _float_or_none(raw.get('P1'))
        self._f0 = _float_or_none(raw.get('F0'))
        self._f1 = _float_or_none(raw.get('F1'))

        self.raj = raw.get('RAJ')
        self.decj = raw.get('DECJ')
        self.dm = _float_or_none(raw.get('DM'))

        self.dist = self._best_distance()

        self.assoc = raw.get('ASSOC')

        # Type (e.g. HE, NRAD)
        self.type = raw.get('TYPE')

    @property
    def period(self) -> Optional[float]:
        """Spin period [s], converting from F0 if needed."""
        if self._p0 is not None:
            return self._p0
        if self._f0 is not None and self._f0 > 0:
            return 1.0 / self._f0
        return None

    @property
    def p_dot(self) -> Optional[float]:
        """Period derivative [s/s], converting from F1/F0 if needed."""
        if self._p1 is not None:
            return self._p1
        if self._f1 is not None and self._f0 is not None and self._f0 > 0:
            return -self._f1 / (self._f0 ** 2)
        return None

    @property
    def f0(self) -> Optional[float]:
        """Spin frequency [Hz], converting from P0 if needed."""
        if self._f0 is not None:
            return self._f0
        if self._p0 is not None and self._p0 > 0:
            return 1.0 / self._p0
        return None

    @property
    def f1(self) -> Optional[float]:
        """Frequency derivative [Hz/s], converting from P0/P1 if needed."""
        if self._f1 is not None:
            return self._f1
        if self._p0 is not None and self._p1 is not None and self._p0 > 0:
            return -self._p1 / (self._p0 ** 2)
        return None

    @property
    def e_dot(self) -> Optional[float]:
        """Spin-down luminosity [erg/s], using canonical I = 10^45 g cm^2."""
        p = self.period
        pdot = self.p_dot
        if p is not None and pdot is not None and p > 0:
            return 4.0 * np.pi**2 * 1e45 * pdot / p**3
        return None

    @property
    def tau_c(self) -> Optional[float]:
        """Characteristic age P/(2*Pdot) [yr]."""
        p = self.period
        pdot = self.p_dot
        if p is not None and pdot is not None and pdot > 0:
            return p / (2.0 * pdot) / (365.25 * 86400.0)
        return None

    @property
    def snr_name(self) -> Optional[str]:
        """SNR association name, if any."""
        if self.assoc is None:
            return None
        for part in self.assoc.split(','):
            part = part.strip()
            if part.startswith('SNR:'):
                name = part[4:]
                if '[' in name:
                    name = name[:name.index('[')]
                return name
        return None

    def _best_distance(self) -> Optional[float]:
        """Best distance estimate [kpc]."""
        d = _float_or_none(self.raw.get('DIST'))
        if d is not None:
            return d
        # AMX/AMN: association max/min distance
        amx = _float_or_none(self.raw.get('DIST_AMX'))
        amn = _float_or_none(self.raw.get('DIST_AMN'))
        if amx is not None and amn is not None:
            return 0.5 * (amx + amn)
        if amx is not None:
            return amx
        if amn is not None:
            return amn
        # YMW16 preferred over CL02
        d_dm = _float_or_none(self.raw.get('DIST_DM'))
        if d_dm is not None:
            return d_dm
        d_dm1 = _float_or_none(self.raw.get('DIST_DM1'))
        if d_dm1 is not None:
            return d_dm1
        return None

    def __repr__(self) -> str:
        parts = [f"PulsarEntry({self.jname}"]
        p = self.period
        pdot = self.p_dot
        if p is not None:
            parts.append(f", P0={p:.4g} s")
        if pdot is not None:
            parts.append(f", P1={pdot:.3e} s/s")
        snr = self.snr_name
        if snr is not None:
            parts.append(f", assoc=SNR:{snr}")
        parts.append(')')
        return ''.join(parts)


class PulsarCatalog:
    """ATNF Pulsar Catalogue interface.

    Parameters
    ----------
    db_path : str, optional
        Path to psrcat.db file.  Defaults to the copy shipped with
        viper.

    Examples
    --------
    >>> cat = PulsarCatalog()
    >>> cat['J0007+7303'].period
    0.31586...
    >>> cat['B0531+21'].snr_name
    'Crab'
    >>> cat.search_snr('CTA')
    [PulsarEntry(J0007+7303, ...)]
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        if db_path is None:
            db_path = _DEFAULT_DB
        if not os.path.exists(db_path):
            raise FileNotFoundError(
                f"Catalogue file not found: {db_path}. "
                f"Expected at {_DEFAULT_DB}"
            )
        self._entries: dict[str, PulsarEntry] = {}
        self._bname_map: dict[str, str] = {}
        self._parse(db_path)
        logger.info("Loaded %d pulsars from %s", len(self._entries), db_path)

    def _parse(self, path: str) -> None:
        """Parse the psrcat.db file."""
        current: dict[str, str] = {}
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                # Entry delimiter
                if line.startswith('@'):
                    if current and 'PSRJ' in current:
                        entry = PulsarEntry(current)
                        self._entries[entry.jname] = entry
                        if entry.bname:
                            self._bname_map[entry.bname] = entry.jname
                    current = {}
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0]
                    value = parts[1]
                    # first occurrence wins
                    if key not in current:
                        current[key] = value

        # last entry may lack '@'
        if current and 'PSRJ' in current:
            entry = PulsarEntry(current)
            self._entries[entry.jname] = entry
            if entry.bname:
                self._bname_map[entry.bname] = entry.jname

    def __getitem__(self, name: str) -> PulsarEntry:
        """Look up a pulsar by J-name or B-name.

        Names are matched with or without the leading 'PSR ' prefix and
        with or without the leading 'J'/'B' letter.
        """
        name = name.strip()
        if name.upper().startswith('PSR '):
            name = name[4:].strip()
        if name.upper().startswith('PSR'):
            name = name[3:].strip()

        if name in self._entries:
            return self._entries[name]

        if not name.startswith('J') and not name.startswith('B'):
            jname = 'J' + name
            if jname in self._entries:
                return self._entries[jname]

        if name in self._bname_map:
            return self._entries[self._bname_map[name]]
        if name.startswith('B') and name[1:] in self._bname_map:
            return self._entries[self._bname_map[name[1:]]]
        if not name.startswith('B'):
            bname = 'B' + name
            if bname in self._bname_map:
                return self._entries[self._bname_map[bname]]

        raise KeyError(
            f"Pulsar '{name}' not found in catalogue. "
            f"Try the full J-name (e.g. 'J0007+7303')."
        )

    def __contains__(self, name: str) -> bool:
        try:
            self[name]
            return True
        except KeyError:
            return False

    def __len__(self) -> int:
        return len(self._entries)

    def search_snr(self, pattern: str) -> list[PulsarEntry]:
        """Find pulsars associated with SNRs matching a pattern.

        Parameters
        ----------
        pattern : str
            Case-insensitive substring to match in SNR association.
            E.g. 'CTA', 'Crab', 'G54', 'G119'.

        Returns
        -------
        list of PulsarEntry
            Matching entries.
        """
        pattern_lower = pattern.lower()
        results = []
        for entry in self._entries.values():
            if entry.assoc and 'SNR:' in entry.assoc:
                if pattern_lower in entry.assoc.lower():
                    results.append(entry)
        return results

    def with_snr(self) -> list[PulsarEntry]:
        """Return all pulsars with an SNR association."""
        return [e for e in self._entries.values() if e.snr_name is not None]

    @property
    def entries(self) -> dict[str, PulsarEntry]:
        """Dict of all entries keyed by J-name."""
        return dict(self._entries)


# ======================================================================
# Utility
# ======================================================================

def _float_or_none(val) -> Optional[float]:
    """Convert a string to float, returning None on failure."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None
