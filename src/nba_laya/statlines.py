"""Stat-line calls: at the end of Q1, halftime and Q3, for the stars in the game.

Per star:  pts_bucket (score), thirty_plus (noul), triple_double (noul, only
when plausible), and a per-game top_scorer (choice among the stars).
Labels come from the final boxscore. Context comes from earlier games only.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from nba_laya.actions import parse_actions
from nba_laya.stats import StatTracker, boxscore_lines, boxscore_starters

CHECKPOINTS = (1, 2, 3)  # period ends
STARS_PER_TEAM = 3
PTS_LEVELS = ["under 10", "10-19", "20-29", "30-39", "40 or more"]
PTS_EDGES = [10, 20, 30, 40]
PRIOR_GAMES = 8
TD_MIN_AVG = 6.0  # ask triple_double only if two of pts/reb/ast average this or more


# ---------------------------------------------------------------- season averages

@dataclass
class PlayerSeason:
    gp: int = 0
    pts: float = 0.0
    reb: float = 0.0
    ast: float = 0.0
    min: float = 0.0
    fga: float = 0.0
    tpa: float = 0.0

    def avg(self) -> dict[str, float]:
        if not self.gp:
            return {}
        return {k: getattr(self, k) / self.gp for k in ("pts", "reb", "ast", "min", "fga", "tpa")}


class PlayerLedger:
    """Season-to-date averages per player, shrunk toward last season early on."""

    def __init__(self) -> None:
        self.season: int | None = None
        self.cur: dict[str, PlayerSeason] = {}
        self.prior: dict[str, dict[str, float]] = {}

    def roll(self, season: int) -> None:
        if self.season is not None and season != self.season:
            self.prior = {n: {**p.avg(), "gp": p.gp} for n, p in self.cur.items() if p.gp}
            self.cur = {}
        self.season = season

    def profile(self, name: str) -> dict | None:
        p = self.cur.get(name)
        cur = p.avg() if p else {}
        pri = self.prior.get(name, {})
        if not cur and not pri:
            return None
        n = p.gp if p else 0
        out = {}
        for k in ("pts", "reb", "ast", "min", "fga", "tpa"):
            c, q = cur.get(k), pri.get(k)
            if c is None and q is None:
                continue
            v = c if q is None else q if c is None else (n * c + PRIOR_GAMES * q) / (n + PRIOR_GAMES)
            out[k] = round(v, 1)
        out["gp"] = n
        return out

    def add(self, final_lines: dict[str, dict]) -> None:
        for name, line in final_lines.items():
            if not line.get("played"):
                continue
            p = self.cur.setdefault(name, PlayerSeason())
            p.gp += 1
            for k in ("pts", "reb", "ast", "min", "fga", "tpa"):
                setattr(p, k, getattr(p, k) + float(line.get(k, 0)))


# ---------------------------------------------------------------- rows

def pts_bucket(points: int) -> int:
    for i, edge in enumerate(PTS_EDGES):
        if points < edge:
            return i
    return len(PTS_EDGES)


def pick_stars(lines_now: dict[str, dict], profiles: dict[str, dict], teams: tuple[str, str]) -> list[str]:
    """Top STARS_PER_TEAM per team by season scoring among players who have appeared tonight."""
    stars = []
    for team in teams:
        cands = [n for n, l in lines_now.items() if l["team"] == team and (l["min"] > 0 or l["pts"] > 0)]
        cands.sort(key=lambda n: -(profiles.get(n, {}).get("pts", 0.0) * 10 + lines_now[n]["pts"]))
        stars += cands[:STARS_PER_TEAM]
    return stars


def player_state(game_ctx: dict, name: str, line: dict, profile: dict | None, period_done: int) -> dict:
    state = {
        **game_ctx,
        "after_period": period_done,
        "player": name,
        "team": line["team"],
        "tonight": {k: line[k] for k in ("pts", "fg", "3p", "ft", "reb", "ast", "pf", "min")},
        "on_court_now": line["on_court"],
    }
    if profile:
        state["season_avg"] = {k: profile[k] for k in ("pts", "reb", "ast", "min", "fga") if k in profile}
        state["season_gp"] = profile.get("gp", 0)
    return state


def player_questions(name: str, profile: dict | None) -> dict:
    qs = {
        "pts_bucket": {
            "type": "score",
            "instructions": f"How many points does {name} finish tonight's game with?",
            "criteria": list(PTS_LEVELS),
        },
        "thirty_plus": {
            "type": "noul",
            "instructions": f"Does {name} finish with 30 or more points?",
            "labels": {"true": "A", "false": "B"},
        },
    }
    if profile and sum(1 for k in ("pts", "reb", "ast") if profile.get(k, 0) >= TD_MIN_AVG) >= 2:
        qs["triple_double"] = {
            "type": "noul",
            "instructions": f"Does {name} finish with a triple-double (10+ points, rebounds and assists)?",
            "labels": {"true": "A", "false": "B"},
        }
    return qs


def player_gold(final: dict, questions: dict) -> dict:
    pts, reb, ast = final["pts"], final["reb"], final["ast"]
    gold = {}
    b = pts_bucket(pts)
    gold["pts_bucket"] = {"type": "score", "label": b,
                          "probabilities": {str(i): (1.0 if i == b else 0.0) for i in range(len(PTS_LEVELS))}}
    t = pts >= 30
    gold["thirty_plus"] = {"type": "noul", "label": "true" if t else "false", "noul": float(t),
                           "probabilities": {"true": float(t), "false": float(not t)}}
    if "triple_double" in questions:
        td = pts >= 10 and reb >= 10 and ast >= 10
        gold["triple_double"] = {"type": "noul", "label": "true" if td else "false", "noul": float(td),
                                 "probabilities": {"true": float(td), "false": float(not td)}}
    return gold


# ---------------------------------------------------------------- baseline

def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def project(line: dict, profile: dict | None, period_done: int, margin: int) -> tuple[float, float]:
    """Expected final points and sd: tonight so far + season rate over expected remaining minutes."""
    played = line["min"]
    if profile and profile.get("min"):
        season_min, ppm = profile["min"], profile["pts"] / max(1.0, profile["min"])
    else:
        season_min, ppm = 28.0, (line["pts"] / played if played else 0.45)
    frac_left = (4 - period_done) / 4
    remaining = max(0.0, season_min * frac_left)
    # Blowouts shorten the night for starters.
    if abs(margin) >= 20 and period_done >= 2:
        remaining *= 0.5
    elif abs(margin) >= 15 and period_done == 3:
        remaining *= 0.6
    if not line["on_court"] and line["pf"] >= 5:
        remaining *= 0.7
    mean = line["pts"] + ppm * remaining
    sd = max(3.0, 0.45 * math.sqrt(max(1.0, ppm * remaining)) * 3.0)
    return mean, sd


def baseline_answers(state: dict, questions: dict, profile: dict | None) -> dict:
    line = state["tonight"] | {"on_court": state["on_court_now"]}
    margin = abs(list(state["score"].values())[0] - list(state["score"].values())[1])
    mean, sd = project(line, profile, state["after_period"], margin)
    out = {}
    if "pts_bucket" in questions:
        edges = [-1e9] + PTS_EDGES + [1e9]
        probs = {}
        for i in range(len(PTS_LEVELS)):
            lo, hi = edges[i], edges[i + 1]
            p = _norm_cdf((hi - 0.5 - mean) / sd) - _norm_cdf((lo - 0.5 - mean) / sd)
            probs[str(i)] = max(p, 1e-4)
        z = sum(probs.values()); probs = {k: round(v / z, 4) for k, v in probs.items()}
        out["pts_bucket"] = {"type": "score", "score": round(sum(int(k) * v for k, v in probs.items()), 3),
                             "probabilities": probs, "legend": {str(i): l for i, l in enumerate(PTS_LEVELS)}}
    if "thirty_plus" in questions:
        p = 1 - _norm_cdf((29.5 - mean) / sd)
        p = min(0.999, max(0.001, p))
        out["thirty_plus"] = {"type": "noul", "noul": round(p, 4), "probabilities": {"A": round(p, 4), "B": round(1 - p, 4)}}
    if "triple_double" in questions:
        p = 0.02
        if profile:
            frac_left = (4 - state["after_period"]) / 4
            parts = []
            for k in ("pts", "reb", "ast"):
                rate = profile[k] / max(1.0, profile["min"])
                m = line[k] + rate * max(0.0, profile["min"] * frac_left)
                s = max(1.5, 0.9 * math.sqrt(max(1.0, m)))
                parts.append(1 - _norm_cdf((9.5 - m) / s))
            p = parts[0] * parts[1] * parts[2]
        p = min(0.98, max(0.005, p))
        out["triple_double"] = {"type": "noul", "noul": round(p, 4), "probabilities": {"A": round(p, 4), "B": round(1 - p, 4)}}
    return out


def top_scorer_answer(projections: dict[str, float]) -> dict:
    # Softmax over projected points; a 4-point edge is roughly 2:1.
    weights = {n: math.exp(m / 6.0) for n, m in projections.items()}
    z = sum(weights.values())
    probs = {n: round(w / z, 4) for n, w in weights.items()}
    best = max(probs, key=probs.get)
    return {"type": "choice", "choice": best, "probabilities": probs, "confidence": probs[best]}


# ---------------------------------------------------------------- per game

def build_game_rows(game_id: str, pbp: dict, box: dict, ledger: PlayerLedger, game_ctx: dict, date: str) -> list[dict]:
    """Rows for one finished game. Caller adds the game to the ledger afterwards."""
    final = boxscore_lines(box)
    actions = parse_actions(pbp["game"]["actions"])
    tracker = StatTracker(boxscore_starters(box))
    teams = tuple(boxscore_starters(box).keys())
    profiles = {n: ledger.profile(n) for n in final}
    rows = []
    for a in actions:
        tracker.apply(a)
        if not a.is_period_end or a.period not in CHECKPOINTS:
            continue
        lines = tracker.snapshot_lines()
        score = {t: 0 for t in teams}
        for n, l in lines.items():
            score[l["team"]] = score.get(l["team"], 0) + l["pts"]
        ctx = {**game_ctx, "score": score}
        stars = pick_stars(lines, {n: p for n, p in profiles.items() if p}, teams)
        projections = {}
        for name in stars:
            line, profile = lines[name], profiles.get(name)
            if name not in final:
                continue
            state = player_state(ctx, name, line, profile, a.period)
            questions = player_questions(name, profile)
            gold = player_gold(final[name], questions)
            answers = baseline_answers(state, questions, profile)
            margin = abs(score[teams[0]] - score[teams[1]])
            projections[name] = project(line, profile, a.period, margin)[0]
            rows.append({
                "game_id": game_id, "game_date": date, "checkpoint": a.period, "kind": "player",
                "model": "rules", "state": state, "questions": questions, "answers": answers,
                "labels": {q: (g["label"] == "true" if g["type"] == "noul" else g["label"]) for q, g in gold.items()},
                "gold": gold,
            })
        if len(projections) >= 2:
            leader = max(stars, key=lambda n: final[n]["pts"]) if stars else None
            cands = {n: f"{n} ({lines[n]['team']}) leads the game in points" for n in projections}
            q = {"top_scorer": {"type": "choice",
                                "instructions": "Which of these players finishes tonight as the game's leading scorer?",
                                "criteria": cands}}
            gold = {"top_scorer": {"type": "choice", "label": leader,
                                   "probabilities": {n: (1.0 if n == leader else 0.0) for n in cands}}}
            state = {**ctx, "after_period": a.period,
                     "candidates": {n: {"tonight": {k: lines[n][k] for k in ("pts", "fg", "min", "pf")},
                                        "season_pts": (profiles.get(n) or {}).get("pts")} for n in projections}}
            rows.append({
                "game_id": game_id, "game_date": date, "checkpoint": a.period, "kind": "game",
                "model": "rules", "state": state, "questions": q,
                "answers": {"top_scorer": top_scorer_answer(projections)},
                "labels": {"top_scorer": leader}, "gold": gold,
            })
    return rows


# ---------------------------------------------------------------- seasons

def build_seasons(data_dir: Path, seasons: list[int], seed_seasons: list[int], out_dir: Path, log=print) -> dict[str, int]:
    """Walk every game with both a tape and a boxscore in date order. Seed seasons
    only feed the ledgers. Writes one JSONL per graded season under out_dir."""
    from nba_laya.context import Ledger as TeamLedger, load_summaries

    dirs = [data_dir / f"season_{s:02d}" for s in sorted(set(seasons + seed_seasons))]
    summaries = load_summaries(dirs, log=log)
    team_ledger = TeamLedger()
    players = PlayerLedger()
    out_dir.mkdir(parents=True, exist_ok=True)
    handles: dict[int, object] = {}
    counts = {"games": 0, "rows": 0, "no_boxscore": 0}
    for g in sorted(summaries.values(), key=lambda s: s.order_key):
        season_dir = data_dir / f"season_{g.season:02d}"
        box_path = season_dir / f"boxscore_{g.game_id}.json"
        pbp_path = season_dir / f"playbyplay_{g.game_id}.json"
        if not box_path.exists():
            counts["no_boxscore"] += 1
            continue
        box = json.loads(box_path.read_text())
        players.roll(g.season)
        if g.season in seasons:
            teams_ctx = team_ledger.pregame(g, g.date, g.season)["teams"]
            home = box["game"]["homeTeam"]["teamTricode"]; away = box["game"]["awayTeam"]["teamTricode"]
            ctx = {"game": f"{away} @ {home}", "teams": {t: {k: teams_ctx[t][k] for k in ("net", "pace") if k in teams_ctx[t]}
                                                        for t in (home, away) if t in teams_ctx}}
            rows = build_game_rows(g.game_id, json.loads(pbp_path.read_text()), box, players, ctx, g.date)
            if g.season not in handles:
                handles[g.season] = (out_dir / f"statlines_{g.season:02d}.jsonl").open("w")
            for row in rows:
                handles[g.season].write(json.dumps(row, separators=(",", ":")) + "\n")
            counts["rows"] += len(rows)
            counts["games"] += 1
            if counts["games"] % 250 == 0:
                log(f"{counts['games']} games, {counts['rows']} rows")
        team_ledger.add(g)
        players.add(boxscore_lines(box))
    for h in handles.values():
        h.close()
    return counts


def export_training(row_paths: list[Path], out_dir: Path, holdout_frac: float = 0.15, calib_frac: float = 0.10) -> dict[str, int]:
    """Stat-line rows -> Laya training files, split by game date."""
    rows = []
    for p in row_paths:
        with p.open() as f:
            rows += [json.loads(line) for line in f]
    games = sorted({(r["game_date"], r["game_id"]) for r in rows})
    n = len(games)
    n_test = int(n * holdout_frac); n_calib = int(n * calib_frac)
    split_of = {}
    for i, g in enumerate(games):
        split_of[g] = "train" if i < n - n_test - n_calib else "calib" if i < n - n_test else "test"
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "calib": 0, "test": 0}
    files = {k: (out_dir / f"{k}.jsonl").open("w") for k in counts}
    for r in rows:
        split = split_of[(r["game_date"], r["game_id"])]
        item = {"id": f"{r['game_id']}-{r['checkpoint']}-{r['kind']}-{r['state'].get('player', 'game')}",
                "game_id": r["game_id"], "workflow": f"statline_{r['kind']}",
                "state": json.dumps(r["state"], separators=(",", ":")),
                "questions": json.dumps(r["questions"], separators=(",", ":")),
                "gold": json.dumps(r["gold"], separators=(",", ":"))}
        files[split].write(json.dumps(item) + "\n")
        counts[split] += 1
    for f in files.values():
        f.close()
    return counts
