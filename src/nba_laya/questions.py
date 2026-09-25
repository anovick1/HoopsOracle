"""The six questions. Every one except swing can be graded from the feed."""

from __future__ import annotations

from nba_laya.state import Snapshot


def build_questions(snapshot: Snapshot) -> dict:
    home, away = snapshot.home, snapshot.away
    questions = {
        "timeout_next2": {
            "type": "noul",
            "instructions": "Will either team call a timeout within the next 2 possessions?",
            "labels": {"true": "A", "false": "B"},
        },
        "next_score": {
            "type": "choice",
            "instructions": "Which team scores next?",
            "criteria": {
                "home": f"{home} scores next",
                "away": f"{away} scores next",
            },
        },
        "sub_next2": {
            "type": "noul",
            "instructions": "Will a player with 4 or more fouls be substituted out within the next 2 possessions?",
            "labels": {"true": "A", "false": "B"},
        },
        "winner": {
            "type": "choice",
            "instructions": "Which team wins this game?",
            "criteria": {
                "home": f"{home} wins",
                "away": f"{away} wins",
            },
        },
        "swing": {
            "type": "score",
            "instructions": "How much did the last possession change the likely outcome of the game?",
            "criteria": ["no real change", "small shift", "meaningful shift", "game changing"],
        },
    }
    if snapshot.run and snapshot.run["points"] > 0:
        team = snapshot.run["team"]
        opp = snapshot.run["opponent"]
        questions["run_continues"] = {
            "type": "noul",
            "instructions": f"Will {team} outscore {opp} over the next 3 possessions?",
            "labels": {"true": "A", "false": "B"},
        }
    return questions
