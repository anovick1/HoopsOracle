"""Which calls are worth posting. Training uses every possession; posting uses this.

Kept out of the model on purpose: thresholds are editorial and change after
preseason, the training labels do not.
"""

from __future__ import annotations

WIN_SWING = 0.08
RUN_POST_POINTS = 8
CLUTCH_SECONDS = 300
CLUTCH_MARGIN = 8
THREE_CALL = 0.60


def is_clutch(snapshot: dict, seconds_left: float) -> bool:
    teams = list(snapshot["score"].values())
    return snapshot["period"] >= 4 and seconds_left <= CLUTCH_SECONDS and abs(teams[0] - teams[1]) <= CLUTCH_MARGIN


def postable(row: dict, previous_wp: float | None, seconds_left: float) -> list[str]:
    """Question ids in this row that clear the posting bar."""
    snap, answers = row["snapshot"], row["answers"]
    picks = []
    wp = row["win_prob_home"]
    if "winner" in answers and previous_wp is not None and abs(wp - previous_wp) >= WIN_SWING:
        picks.append("winner")
    run = snap.get("run")
    if "run_continues" in answers and run and run["points"] >= RUN_POST_POINTS:
        picks.append("run_continues")
    clutch = is_clutch(snap, seconds_left)
    if clutch:
        picks += [q for q in ("shooter", "possession_scores") if q in answers]
    score_type = answers.get("score_type")
    if score_type and (clutch or score_type["probabilities"].get("three", 0) >= THREE_CALL):
        picks.append("score_type")
    if "comeback" in answers:
        picks.append("comeback")
    return picks
