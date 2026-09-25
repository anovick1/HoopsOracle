"""Game clock helpers. Live feed clocks look like PT11M58.00S."""

from __future__ import annotations

import re

_ISO = re.compile(
    r"PT(?:(?P<h>\d+)H)?(?:(?P<m>\d+)M)?(?:(?P<s>\d+(?:\.\d+)?)S)?",
    re.IGNORECASE,
)
_COLON = re.compile(r"^(?:(?P<m>\d+):)?(?P<s>\d+(?:\.\d+)?)$")


def parse_clock(clock: str | None) -> float:
    """Seconds remaining in the period. Unknown clocks return 0."""
    if not clock:
        return 0.0
    text = str(clock).strip()
    iso = _ISO.fullmatch(text)
    if iso:
        hours = float(iso.group("h") or 0)
        minutes = float(iso.group("m") or 0)
        seconds = float(iso.group("s") or 0)
        return hours * 3600 + minutes * 60 + seconds
    colon = _COLON.fullmatch(text)
    if colon:
        minutes = float(colon.group("m") or 0)
        return minutes * 60 + float(colon.group("s") or 0)
    return 0.0


def display_clock(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 60}:{seconds % 60:02d}"


def period_length(period: int, period_type: str | None) -> int:
    if (period_type or "").upper() == "OVERTIME" or period > 4:
        return 5 * 60
    return 12 * 60


def elapsed_seconds(period: int, clock: str | None, period_type: str | None) -> float:
    """Seconds since opening tip, regulation then overtime."""
    prior = 0
    for p in range(1, max(period, 1)):
        prior += 5 * 60 if p > 4 else 12 * 60
    length = period_length(period, period_type)
    remaining = min(parse_clock(clock), length)
    return prior + (length - remaining)


def seconds_left_in_game(period: int, clock: str | None, period_type: str | None) -> float:
    """Regulation time left, plus the current overtime period if already there.

    Future overtime is not included. A tied game under a minute is treated as
    having only the time still on the clock.
    """
    remaining = parse_clock(clock)
    if period > 4 or (period_type or "").upper() == "OVERTIME":
        return remaining
    return remaining + (4 - period) * 12 * 60
