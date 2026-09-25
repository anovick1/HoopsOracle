"""Rules baseline. Same answer shape as Laya and Jev, so the grader does not care who answered."""

from __future__ import annotations

from nba_laya.state import Snapshot
from nba_laya.wp import swing_bucket


def _noul(probability: float) -> dict:
    p = min(1.0, max(0.0, probability))
    return {"type": "noul", "noul": round(p, 4), "probabilities": {"A": round(p, 4), "B": round(1 - p, 4)}}


def _choice(probabilities: dict[str, float]) -> dict:
    total = sum(probabilities.values()) or 1.0
    probs = {key: round(value / total, 4) for key, value in probabilities.items()}
    choice = max(probs, key=probs.get)
    return {"type": "choice", "choice": choice, "probabilities": probs, "confidence": round(probs[choice], 4)}


def _score(bucket: int) -> dict:
    probs = {str(i): 0.05 for i in range(4)}
    probs[str(bucket)] = 0.85
    return {
        "type": "score",
        "score": float(bucket),
        "probabilities": probs,
        "legend": {"0": "no real change", "1": "small shift", "2": "meaningful shift", "3": "game changing"},
    }


def answer(snapshot: Snapshot, previous_wp: float | None, questions: dict) -> dict:
    answers = {}
    if "timeout_next2" in questions:
        answers["timeout_next2"] = _noul(_p_timeout(snapshot))
    if "run_continues" in questions:
        answers["run_continues"] = _noul(0.45 if snapshot.run and snapshot.run["points"] >= 6 else 0.35)
    if "next_score" in questions:
        home_p = 0.58 if snapshot.possession == snapshot.home else 0.42
        answers["next_score"] = _choice({"home": home_p, "away": 1 - home_p})
    if "sub_next2" in questions:
        answers["sub_next2"] = _noul(_p_sub(snapshot))
    if "winner" in questions:
        answers["winner"] = _choice({"home": snapshot.win_prob_home, "away": 1 - snapshot.win_prob_home})
    if "swing" in questions:
        bucket = swing_bucket(previous_wp, snapshot.win_prob_home)
        answers["swing"] = _score(0 if bucket is None else bucket)
    return answers


def _p_timeout(snapshot: Snapshot) -> float:
    run = snapshot.run["points"] if snapshot.run else 0
    trailing = snapshot.away if snapshot.run and snapshot.run["team"] == snapshot.home else snapshot.home
    if snapshot.run is None:
        trailing = snapshot.away if snapshot.score[snapshot.home] >= snapshot.score[snapshot.away] else snapshot.home
    timeouts = snapshot.timeouts_left.get(trailing, 0)
    p = 0.08
    if run >= 8 and timeouts > 0:
        p = 0.65
    if run >= 12 and timeouts > 0:
        p = 0.80
    margin = abs(snapshot.score[snapshot.home] - snapshot.score[snapshot.away])
    if snapshot.seconds_left < 120 and margin <= 10 and timeouts > 0:
        p = max(p, 0.55)
    if timeouts <= 0:
        p = 0.02
    return p


def _p_sub(snapshot: Snapshot) -> float:
    high = [
        fouls
        for players in snapshot.fouls.values()
        for fouls in players.values()
        if fouls >= 4
    ]
    if any(fouls >= 5 for fouls in high):
        return 0.75
    if high:
        return 0.45
    return 0.05
