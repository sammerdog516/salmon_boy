from __future__ import annotations

BAND_ALIASES = {
    "B2": "B2",
    "B02": "B2",
    "BLUE": "B2",
    "B3": "B3",
    "B03": "B3",
    "GREEN": "B3",
    "B4": "B4",
    "B04": "B4",
    "RED": "B4",
    "B5": "B5",
    "B05": "B5",
    "RED_EDGE": "B5",
    "REDEDGE": "B5",
    "B8": "B8",
    "B08": "B8",
    "NIR": "B8",
}


def normalize_band_name(name: str) -> str:
    key = name.strip().upper().replace("-", "_")
    return BAND_ALIASES.get(key, key)

