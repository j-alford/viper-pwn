"""Pytest configuration: VIPER_CONSTANTS must be set before any viper import."""

import os
os.environ.setdefault('VIPER_CONSTANTS', 'gsz2009')
