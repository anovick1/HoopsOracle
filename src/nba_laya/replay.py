"""Replay one finished game and write a graded JSONL log."""

from __future__ import annotations

import json
from pathlib import Path

from nba_laya.client import SystemOneClient
from nba_laya.feed import fetch_json, load_file, playbyplay_url
from nba_laya.grader import grade
from nba_laya.metrics import SCOREBOARD_QUESTIONS, summarize
from nba_laya.questions import build_questions
from nba_laya.rules import answer
from nba_laya.state import GameWalk, load_game


def replay(payload: dict, model: str = "rules", client: SystemOneClient | None = None, pregame=None) -> list[dict]:
    game_id, actions = load_game(payload)
    decisions = GameWalk(game_id, actions, pregame=pregame).walk()
    grade(decisions, actions)
    rows = []
    previous_wp = None
    for decision in decisions:
        snap = decision.snapshot
        questions = build_questions(snap)
        if model == "rules":
            answers = answer(snap, previous_wp, questions)
        elif client is not None:
            answers = client.system_one(snap.as_state(), questions)
        else:
            raise RuntimeError(f"model {model!r} needs a client")
        previous_wp = snap.win_prob_home
        rows.append({
            "game_id": game_id,
            "game_date": pregame.date if pregame is not None else None,
            "action_number": snap.action_number,
            "model": model,
            "snapshot": snap.as_state(),
            "win_prob_home": round(snap.win_prob_home, 4),
            "questions": questions,
            "answers": answers,
            "labels": decision.labels,
        })
    return rows


def write_log(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def scoreboard(rows: list[dict]) -> list[dict]:
    lines = []
    for question in SCOREBOARD_QUESTIONS:
        summary = summarize(rows, question)
        if summary:
            lines.append(summary)
    return lines


def format_scoreboard(rows: list[dict]) -> str:
    lines = []
    for item in scoreboard(rows):
        lines.append(
            f"{item['question']:<16} n={item['n']:<4} acc={item['accuracy']:.3f}  brier={item['brier']:.3f}  ece={item['ece']:.3f}"
        )
    return "\n".join(lines) if lines else "no graded calls"


def load_payload(game_id: str | None, path: Path | None) -> dict:
    if path is not None:
        return load_file(path)
    if not game_id:
        raise RuntimeError("pass a game id or a play-by-play file")
    payload, _ = fetch_json(playbyplay_url(game_id))
    if payload is None:
        raise RuntimeError(f"empty response for {game_id}")
    return payload
