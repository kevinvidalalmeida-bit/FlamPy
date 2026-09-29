"""Conversion helpers for JSON metadata written by the public API."""

from __future__ import annotations

import math

import numpy as np


def json_safe(value):
    """Convert NumPy values and non-finite floats into JSON-compatible data."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
