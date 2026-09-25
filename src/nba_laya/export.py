"""Turn graded logs into Laya training rows.

One row per possession: the snapshot as ``state``, the question battery as
``questions``, and a ``gold`` distribution per graded question with all the
mass on the truth. This is the shape Convai's fine-tune notebook reads.

Splits are by game id, which is chronological within a season, never by row.
Consecutive possessions are near-duplicates and would leak across a random split.
"""

from __future__ import annotations

import json
from pathlib import Path


def gold_for(question: dict, label) -> dict | None:
    kind = question["type"]
    if kind == "noul":
        p_true = 1.0 if label else 0.0
        return {"type": "noul", "label": "true" if label else "false", "noul": p_true,
                "probabilities": {"true": p_true, "false": 1.0 - p_true}}
    if kind == "choice":
        keys = list(question["criteria"])
        if label not in keys:
            return None
        return {"type": "choice", "label": label,
                "probabilities": {k: (1.0 if k == label else 0.0) for k in keys}}
    if kind == "score":
        levels = len(question["criteria"])
        idx = int(label)
        if not 0 <= idx < levels:
            return None
        return {"type": "score", "label": idx,
                "probabilities": {str(i): (1.0 if i == idx else 0.0) for i in range(levels)}}
    return None


def to_training_row(row: dict, drop: set[str] = frozenset()) -> dict | None:
    questions = {}
    gold = {}
    for qid, question in row["questions"].items():
        if qid in drop or qid not in row["labels"]:
            continue
        g = gold_for(question, row["labels"][qid])
        if g is None:
            continue
        questions[qid] = question
        gold[qid] = g
    if not gold:
        return None
    return {
        "id": f"{row['game_id']}-{row['action_number']}",
        "game_id": row["game_id"],
        "workflow": "nba_possession",
        "state": json.dumps(row["snapshot"], separators=(",", ":")),
        "questions": json.dumps(questions, separators=(",", ":")),
        "gold": json.dumps(gold, separators=(",", ":")),
    }


def export(log_paths: list[Path], out_dir: Path, holdout_frac: float = 0.15, calib_frac: float = 0.10,
           drop: set[str] = frozenset({"swing"})) -> dict[str, int]:
    rows_by_game: dict[str, list[dict]] = {}
    for path in log_paths:
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                rows_by_game.setdefault(row["game_id"], []).append(row)
    games = sorted(rows_by_game)
    n = len(games)
    n_test = max(1, int(n * holdout_frac)) if n > 2 else 0
    n_calib = max(1, int(n * calib_frac)) if n > 3 else 0
    splits = {
        "train": games[: n - n_test - n_calib],
        "calib": games[n - n_test - n_calib : n - n_test],
        "test": games[n - n_test :],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, game_ids in splits.items():
        written = 0
        with (out_dir / f"{name}.jsonl").open("w") as handle:
            for gid in game_ids:
                for row in rows_by_game[gid]:
                    item = to_training_row(row, drop)
                    if item is not None:
                        handle.write(json.dumps(item) + "\n")
                        written += 1
        counts[name] = written
        counts[f"{name}_games"] = len(game_ids)
    return counts
