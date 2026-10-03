"""Which source answers which question.

Decided on held-out games, not by preference. Laya beats the formulas on
run_continues, possession_scores and score_type and ties on shooter; the
clock formula beats it on winner, and comeback has too few cases to trust
the model. Re-run `nba-laya scoreboard` on new logs before changing this.
"""

from __future__ import annotations

from nba_laya.rules import answer as rules_answer

SOURCES = {
    "winner": "rules",
    "comeback": "rules",
    "run_continues": "laya",
    "possession_scores": "laya",
    "score_type": "laya",
    "shooter": "laya",
}


def hybrid_answer(snapshot, previous_wp, questions: dict, laya_answers: dict) -> dict:
    """Merge: rules for the questions the formula wins, Laya for the rest."""
    rules = rules_answer(snapshot, previous_wp, questions)
    out = {}
    for qid in questions:
        source = SOURCES.get(qid, "laya")
        if source == "rules" or qid not in laya_answers:
            if qid in rules:
                out[qid] = dict(rules[qid], source="rules")
        else:
            out[qid] = dict(laya_answers[qid], source="laya")
    return out
