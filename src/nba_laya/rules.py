"""Rules baseline. Same answer shape as Laya and Jev, so the grader does not care who answered.

Each rule is the dumbest defensible answer. The model has to beat these.
"""

from __future__ import annotations

from nba_laya.state import Snapshot

# Base rates from the first 986 games of 2025-26 (the training split), rounded.
# Test-split games are excluded so the baseline never sees the games it is scored on.
P_POSSESSION_SCORES = 0.50
SCORE_TYPE_MIX = {"two": 0.533, "three": 0.239, "free_throw": 0.228}
P_RUN_CONTINUES = 0.33


def _noul(probability: float) -> dict:
    p = min(0.999, max(0.001, probability))
    return {"type": "noul", "noul": round(p, 4), "probabilities": {"A": round(p, 4), "B": round(1 - p, 4)}}


def _choice(probabilities: dict[str, float]) -> dict:
    total = sum(probabilities.values()) or 1.0
    probs = {key: round(value / total, 4) for key, value in probabilities.items()}
    choice = max(probs, key=probs.get)
    return {"type": "choice", "choice": choice, "probabilities": probs, "confidence": round(probs[choice], 4)}


def answer(snapshot: Snapshot, previous_wp: float | None, questions: dict) -> dict:
    answers = {}
    wp_home = snapshot.win_prob_home
    if "winner" in questions:
        answers["winner"] = _choice({"home": wp_home, "away": 1 - wp_home})
    if "possession_scores" in questions:
        answers["possession_scores"] = _noul(P_POSSESSION_SCORES)
    if "score_type" in questions:
        answers["score_type"] = _choice(dict(SCORE_TYPE_MIX))
    if "run_continues" in questions:
        answers["run_continues"] = _noul(P_RUN_CONTINUES)
    if "shooter" in questions:
        # Shots so far tonight, plus one so a player with zero still has a chance.
        answers["shooter"] = _choice({
            name: snapshot.shots_tonight.get(name, 0) + 1 for name in questions["shooter"]["criteria"]
        })
    if "comeback" in questions:
        answers["comeback"] = _noul(_p_comeback(snapshot))
    return answers


def _p_comeback(snapshot: Snapshot) -> float:
    """Reflection principle: for a driftless margin, P(ever ahead) is about 2 x P(ahead at the end)."""
    team = snapshot.comeback["team"]
    wp_team = snapshot.win_prob_home if team == snapshot.home else 1 - snapshot.win_prob_home
    return min(0.95, 2 * wp_team)
