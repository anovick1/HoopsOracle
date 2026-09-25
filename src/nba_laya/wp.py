"""A transparent win-probability estimate. Code owns this, not the model."""

from __future__ import annotations

import math


def win_prob(home_margin: float, seconds_left: float, home_has_ball: bool) -> float:
    """Logistic of margin over a scale that shrinks with the clock.

    Remaining-score noise is about 11 points across a full game, shrinking
    with the square root of time left. Possession is worth roughly 0.7 points.
    This is a baseline for grading ``swing``, not a forecast.
    """
    minutes_left = max(seconds_left, 1.0) / 60.0
    sigma = 11.0 * math.sqrt(minutes_left / 48.0)
    bonus = 0.7 if home_has_ball else -0.7
    z = (home_margin + bonus) / max(sigma, 0.5)
    return 1.0 / (1.0 + math.exp(-z))


def swing_bucket(previous: float | None, current: float) -> int | None:
    if previous is None:
        return None
    delta = abs(current - previous)
    if delta < 0.01:
        return 0
    if delta < 0.03:
        return 1
    if delta < 0.08:
        return 2
    return 3
