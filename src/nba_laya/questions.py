"""The question battery. Every question is graded from the play-by-play.

Some questions only exist in some states: run_continues needs a 6+ run,
shooter needs all five on the floor for the team with the ball, comeback
fires once when a team first falls 10 behind.
"""

from __future__ import annotations

from nba_laya.state import Snapshot

SCORE_TYPES = {
    "two": "a two-point field goal",
    "three": "a three-pointer",
    "free_throw": "a free throw",
}


def build_questions(snapshot: Snapshot) -> dict:
    home, away, ball = snapshot.home, snapshot.away, snapshot.possession
    questions = {
        "winner": {
            "type": "choice",
            "instructions": "Which team wins this game?",
            "criteria": {"home": f"{home} wins", "away": f"{away} wins"},
        },
        "possession_scores": {
            "type": "noul",
            "instructions": f"Will {ball} score on this possession?",
            "labels": {"true": "A", "false": "B"},
        },
        "score_type": {
            "type": "choice",
            "instructions": "What kind of basket is the next score in this game?",
            "criteria": dict(SCORE_TYPES),
        },
    }
    if snapshot.run and snapshot.run["points"] > 0:
        team, opp = snapshot.run["team"], snapshot.run["opponent"]
        questions["run_continues"] = {
            "type": "noul",
            "instructions": f"{team} is on a {snapshot.run['points']}-0 run. Will {team} outscore {opp} over the next 3 possessions?",
            "labels": {"true": "A", "false": "B"},
        }
    menu = snapshot.shooter_menu
    if menu:
        questions["shooter"] = {
            "type": "choice",
            "instructions": f"Who takes {ball}'s next shot attempt?",
            "criteria": {name: f"{name} shoots next" for name in menu},
        }
    if snapshot.comeback:
        team, opp, deficit = (snapshot.comeback[k] for k in ("team", "opponent", "deficit"))
        questions["comeback"] = {
            "type": "noul",
            "instructions": f"{team} trails {opp} by {deficit}. Will {team} take the lead at any point before the game ends?",
            "labels": {"true": "A", "false": "B"},
        }
    return questions
