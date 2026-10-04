"""Stat-line calls: at the end of Q1, halftime and Q3, for the stars in the game.

Per star:  pts_bucket (score), thirty_plus (noul), triple_double (noul, only
when plausible), and a per-game top_scorer (choice among the stars).
Labels come from the final boxscore. Everything else comes from earlier games
(player game logs, team ratings) or from tonight's tape up to the checkpoint.

Features per star at a checkpoint:
  tonight       pts/fg/3p/ft/reb/ast/pf/min so far, on court now, minutes pace vs usual
  season        avg pts/reb/ast/min/fga, games played, pts std (consistency)
  form          last 5 and last 10 games: pts, min
  matchup       home/away, rest days, back-to-back, opponent def rating and pace
  teammates_out usual rotation players with zero minutes so far, and their scoring
  projection    the formula's expected final points and sd (so a model can start from it)
"""

from __future__ import annotations

import json
import math
import statistics
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from nba_laya.actions import parse_actions
from nba_laya.stats import StatTracker, boxscore_lines, boxscore_starters

CHECKPOINTS = (1, 2, 3)  # period ends
STARS_PER_TEAM = 3
PTS_LEVELS = ["under 10", "10-19", "20-29", "30-39", "40 or more"]
PTS_EDGES = [10, 20, 30, 40]
PRIOR_GAMES = 8
TD_MIN_AVG = 6.0
ROTATION_MPG = 18.0      # a usual rotation player averages at least this over the team's last 10
ROTATION_WINDOW = 10
FORM_WINDOW = 10


# ---------------------------------------------------------------- player ledger

@dataclass
class GameLog:
    date: str
    pts: int
    reb: int
    ast: int
    min: float
    fga: int
    tpa: int
    fta: int
    starter: bool


class PlayerLedger:
    """Game logs per player this season, with last season's averages as a prior."""

    def __init__(self) -> None:
        self.season: int | None = None
        self.logs: dict[str, list[GameLog]] = {}
        self.prior: dict[str, dict[str, float]] = {}
        # team -> deque of {name: minutes} for the team's last ROTATION_WINDOW games
        self.team_minutes: dict[str, deque] = {}

    def roll(self, season: int) -> None:
        if self.season is not None and season != self.season:
            self.prior = {n: self._avg(logs) | {"gp": len(logs)} for n, logs in self.logs.items() if logs}
            self.logs = {}
            self.team_minutes = {}
        self.season = season

    @staticmethod
    def _avg(logs: list[GameLog]) -> dict[str, float]:
        if not logs:
            return {}
        n = len(logs)
        out = {k: sum(getattr(g, k) for g in logs) / n for k in ("pts", "reb", "ast", "min", "fga", "tpa", "fta")}
        out["pts_sd"] = statistics.pstdev([g.pts for g in logs]) if n > 1 else max(4.0, 0.4 * out["pts"])
        out["start_rate"] = sum(g.starter for g in logs) / n
        return out

    def profile(self, name: str) -> dict | None:
        logs = self.logs.get(name, [])
        cur = self._avg(logs)
        pri = self.prior.get(name, {})
        if not cur and not pri:
            return None
        n = len(logs)
        out = {}
        for k in ("pts", "reb", "ast", "min", "fga", "tpa", "fta", "pts_sd", "start_rate"):
            c, q = cur.get(k), pri.get(k)
            if c is None and q is None:
                continue
            v = c if q is None else q if c is None else (n * c + PRIOR_GAMES * q) / (n + PRIOR_GAMES)
            out[k] = round(v, 2)
        out["gp"] = n
        if logs:
            last5, last10 = logs[-5:], logs[-FORM_WINDOW:]
            out["l5_pts"] = round(sum(g.pts for g in last5) / len(last5), 1)
            out["l5_min"] = round(sum(g.min for g in last5) / len(last5), 1)
            out["l10_pts"] = round(sum(g.pts for g in last10) / len(last10), 1)
            out["l10_min"] = round(sum(g.min for g in last10) / len(last10), 1)
        return out

    def usual_rotation(self, team: str) -> dict[str, float]:
        """name -> minutes per game over the team's last games, for players above ROTATION_MPG."""
        games = self.team_minutes.get(team)
        if not games:
            return {}
        totals: dict[str, float] = {}
        for g in games:
            for name, m in g.items():
                totals[name] = totals.get(name, 0.0) + m
        return {n: round(t / len(games), 1) for n, t in totals.items() if t / len(games) >= ROTATION_MPG}

    def add(self, final_lines: dict[str, dict], date: str) -> None:
        by_team: dict[str, dict[str, float]] = {}
        for name, line in final_lines.items():
            if not line.get("played"):
                continue
            self.logs.setdefault(name, []).append(GameLog(
                date=date, pts=int(line["pts"]), reb=int(line["reb"]), ast=int(line["ast"]), min=float(line["min"]),
                fga=int(line["fga"]), tpa=int(line["tpa"]), fta=int(line["fta"]), starter=bool(line["starter"]),
            ))
            by_team.setdefault(line["team"], {})[name] = float(line["min"])
        for team, mins in by_team.items():
            self.team_minutes.setdefault(team, deque(maxlen=ROTATION_WINDOW)).append(mins)


# ---------------------------------------------------------------- questions and labels

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
        cands.sort(key=lambda n: -((profiles.get(n) or {}).get("pts", 0.0) * 10 + lines_now[n]["pts"]))
        stars += cands[:STARS_PER_TEAM]
    return stars


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
    b = pts_bucket(pts)
    gold = {"pts_bucket": {"type": "score", "label": b,
                           "probabilities": {str(i): (1.0 if i == b else 0.0) for i in range(len(PTS_LEVELS))}}}
    t = pts >= 30
    gold["thirty_plus"] = {"type": "noul", "label": "true" if t else "false", "noul": float(t),
                           "probabilities": {"true": float(t), "false": float(not t)}}
    if "triple_double" in questions:
        td = pts >= 10 and reb >= 10 and ast >= 10
        gold["triple_double"] = {"type": "noul", "label": "true" if td else "false", "noul": float(td),
                                 "probabilities": {"true": float(td), "false": float(not td)}}
    return gold


# ---------------------------------------------------------------- projection formula

def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def expected_minutes_so_far(usual_min: float, period_done: int) -> float:
    return usual_min * period_done / 4


def project(tonight: dict, profile: dict | None, period_done: int, margin: int, vacancy_pts: float) -> tuple[float, float]:
    """Expected final points and sd.

    Remaining minutes: the player's usual share of the rest of the game, trimmed in
    blowouts and when he is sitting with fouls. Scoring rate: season rate nudged by
    tonight's rate and by teammates' missing scoring (usage vacancy).
    """
    played = tonight["min"]
    if profile and profile.get("min"):
        usual_min = profile["min"]
        rate = profile["pts"] / max(1.0, usual_min)
        sd_rate = profile.get("pts_sd", 6.0) / max(1.0, usual_min)
    else:
        usual_min, rate, sd_rate = 28.0, (tonight["pts"] / played if played else 0.45), 0.25
    frac_left = (4 - period_done) / 4
    remaining = max(0.0, usual_min * frac_left)
    # Tonight's minutes pace tells us about the coach's plan (foul trouble, blowout, matchup).
    expected_now = expected_minutes_so_far(usual_min, period_done)
    if expected_now > 4:
        pace_ratio = min(1.3, max(0.4, played / expected_now))
        remaining *= 0.5 + 0.5 * pace_ratio
    if abs(margin) >= 20 and period_done >= 2:
        remaining *= 0.5
    elif abs(margin) >= 15 and period_done == 3:
        remaining *= 0.6
    if tonight["pf"] >= 5:
        remaining *= 0.7
    # Usage vacancy: roughly a third of missing teammates' points flow to the stars who play.
    rate_tonight = tonight["pts"] / played if played >= 6 else rate
    rate_blend = 0.75 * rate + 0.25 * rate_tonight
    rate_blend += 0.33 * vacancy_pts / max(1.0, usual_min) / 3.0
    mean = tonight["pts"] + rate_blend * remaining
    sd = max(2.5, sd_rate * remaining * 0.9 + 1.5)
    return mean, sd


def baseline_answers(state: dict, questions: dict, profile: dict | None) -> dict:
    mean, sd = state["projection"]["pts_mean"], state["projection"]["pts_sd"]
    out = {}
    if "pts_bucket" in questions:
        edges = [-1e9] + PTS_EDGES + [1e9]
        probs = {}
        for i in range(len(PTS_LEVELS)):
            lo, hi = edges[i], edges[i + 1]
            probs[str(i)] = max(_norm_cdf((hi - 0.5 - mean) / sd) - _norm_cdf((lo - 0.5 - mean) / sd), 1e-4)
        z = sum(probs.values()); probs = {k: round(v / z, 4) for k, v in probs.items()}
        out["pts_bucket"] = {"type": "score", "score": round(sum(int(k) * v for k, v in probs.items()), 3),
                             "probabilities": probs, "legend": {str(i): l for i, l in enumerate(PTS_LEVELS)}}
    if "thirty_plus" in questions:
        p = min(0.999, max(0.001, 1 - _norm_cdf((29.5 - mean) / sd)))
        out["thirty_plus"] = {"type": "noul", "noul": round(p, 4), "probabilities": {"A": round(p, 4), "B": round(1 - p, 4)}}
    if "triple_double" in questions:
        p = 0.02
        if profile and profile.get("min"):
            frac_left = (4 - state["after_period"]) / 4
            tonight = state["tonight"]
            parts = []
            for k in ("pts", "reb", "ast"):
                rate = profile[k] / max(1.0, profile["min"])
                m = tonight[k] + rate * max(0.0, profile["min"] * frac_left)
                s = max(1.5, 0.9 * math.sqrt(max(1.0, m)))
                parts.append(1 - _norm_cdf((9.5 - m) / s))
            p = parts[0] * parts[1] * parts[2]
        p = min(0.98, max(0.005, p))
        out["triple_double"] = {"type": "noul", "noul": round(p, 4), "probabilities": {"A": round(p, 4), "B": round(1 - p, 4)}}
    return out


def top_scorer_answer(projections: dict[str, tuple[float, float]]) -> dict:
    """P(each candidate finishes highest), by Monte Carlo over the projections."""
    import random

    rng = random.Random(7)
    names = list(projections)
    wins = {n: 0 for n in names}
    for _ in range(2000):
        draws = {n: rng.gauss(*projections[n]) for n in names}
        wins[max(draws, key=draws.get)] += 1
    probs = {n: round(max(w, 1) / 2000, 4) for n, w in wins.items()}
    z = sum(probs.values()); probs = {n: round(p / z, 4) for n, p in probs.items()}
    best = max(probs, key=probs.get)
    return {"type": "choice", "choice": best, "probabilities": probs, "confidence": probs[best]}


# ---------------------------------------------------------------- state

def _rest(team_ctx: dict) -> tuple[int | None, bool]:
    rest = team_ctx.get("rest")
    return rest, (rest is not None and rest <= 1)


def player_state(game_ctx: dict, name: str, line: dict, profile: dict | None, period_done: int,
                 teammates_out: dict[str, float], projection: tuple[float, float]) -> dict:
    team = line["team"]
    opp = game_ctx["away"] if team == game_ctx["home"] else game_ctx["home"]
    team_ctx = game_ctx["teams"].get(team, {})
    opp_ctx = game_ctx["teams"].get(opp, {})
    rest, b2b = _rest(team_ctx)
    tonight = {k: line[k] for k in ("pts", "fg", "3p", "ft", "reb", "ast", "pf", "min")}
    state = {
        "game": game_ctx["game"],
        "after_period": period_done,
        "score": game_ctx["score"],
        "player": name,
        "team": team,
        "home": team == game_ctx["home"],
        "tonight": tonight,
        "on_court_now": line["on_court"],
    }
    if profile:
        state["season"] = {k: profile[k] for k in ("pts", "reb", "ast", "min", "fga", "pts_sd", "gp") if k in profile}
        if "l5_pts" in profile:
            state["form"] = {"l5_pts": profile["l5_pts"], "l5_min": profile["l5_min"],
                             "l10_pts": profile["l10_pts"], "l10_min": profile["l10_min"]}
        expected_now = expected_minutes_so_far(profile.get("min", 28.0), period_done)
        state["min_pace"] = round(line["min"] / expected_now, 2) if expected_now > 4 else None
    matchup = {}
    if rest is not None:
        matchup["rest"] = rest
        matchup["b2b"] = b2b
    if opp_ctx:
        matchup["opp_def"] = opp_ctx.get("def")
        matchup["opp_pace"] = opp_ctx.get("pace")
    if matchup:
        state["matchup"] = matchup
    if teammates_out:
        state["teammates_out"] = teammates_out
    state["projection"] = {"pts_mean": round(projection[0], 1), "pts_sd": round(projection[1], 1)}
    return {k: v for k, v in state.items() if v not in (None, {}, [])}


# ---------------------------------------------------------------- per game

def build_game_rows(game_id: str, pbp: dict, box: dict, ledger: PlayerLedger, game_ctx: dict, date: str) -> list[dict]:
    """Rows for one finished game. Caller adds the game to the ledger afterwards."""
    final = boxscore_lines(box)
    actions = parse_actions(pbp["game"]["actions"])
    starters = boxscore_starters(box)
    tracker = StatTracker(starters)
    teams = tuple(starters.keys())
    profiles = {n: ledger.profile(n) for n in final}
    rotation = {t: ledger.usual_rotation(t) for t in teams}
    rows = []
    for a in actions:
        tracker.apply(a)
        if not a.is_period_end or a.period not in CHECKPOINTS:
            continue
        lines = tracker.snapshot_lines()
        score = {t: 0 for t in teams}
        for l in lines.values():
            score[l["team"]] = score.get(l["team"], 0) + l["pts"]
        margin = abs(score[teams[0]] - score[teams[1]])
        ctx = {**game_ctx, "score": score}
        # Usual rotation players with no minutes through this checkpoint are out tonight.
        out_by_team = {}
        for t in teams:
            out = {n: (profiles.get(n) or {}).get("pts", 0.0) for n, mpg in rotation[t].items()
                   if lines.get(n, {}).get("min", 0) == 0 and not lines.get(n, {}).get("on_court")}
            out_by_team[t] = {n: round(p, 1) for n, p in out.items() if p >= 5.0}
        stars = pick_stars(lines, profiles, teams)
        projections: dict[str, tuple[float, float]] = {}
        for name in stars:
            if name not in final:
                continue
            line, profile = lines[name], profiles.get(name)
            vacancy = sum(out_by_team[line["team"]].values())
            proj = project(line, profile, a.period, margin, vacancy)
            projections[name] = proj
            state = player_state(ctx, name, line, profile, a.period, out_by_team[line["team"]], proj)
            questions = player_questions(name, profile)
            gold = player_gold(final[name], questions)
            rows.append({
                "game_id": game_id, "game_date": date, "checkpoint": a.period, "kind": "player",
                "model": "rules", "state": state, "questions": questions,
                "answers": baseline_answers(state, questions, profile),
                "labels": {q: (g["label"] == "true" if g["type"] == "noul" else g["label"]) for q, g in gold.items()},
                "gold": gold,
            })
        if len(projections) >= 2:
            leader = max(projections, key=lambda n: final[n]["pts"])
            cands = {n: f"{n} ({lines[n]['team']}) leads the game in points" for n in projections}
            q = {"top_scorer": {"type": "choice",
                                "instructions": "Which of these players finishes tonight as the game's leading scorer?",
                                "criteria": cands}}
            gold = {"top_scorer": {"type": "choice", "label": leader,
                                   "probabilities": {n: (1.0 if n == leader else 0.0) for n in cands}}}
            state = {"game": ctx["game"], "after_period": a.period, "score": score,
                     "candidates": {n: {"team": lines[n]["team"], "pts": lines[n]["pts"], "fg": lines[n]["fg"],
                                        "min": lines[n]["min"], "pf": lines[n]["pf"],
                                        "season_pts": (profiles.get(n) or {}).get("pts"),
                                        "proj": round(projections[n][0], 1)} for n in projections}}
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
        home = box["game"]["homeTeam"]["teamTricode"]; away = box["game"]["awayTeam"]["teamTricode"]
        if g.season in seasons:
            teams_ctx = team_ledger.pregame(g, g.date, g.season)["teams"]
            ctx = {"game": f"{away} @ {home}", "home": home, "away": away,
                   "teams": {t: {k: teams_ctx[t][k] for k in ("net", "off", "def", "pace", "rest") if k in teams_ctx[t]}
                             for t in (home, away) if t in teams_ctx}}
            rows = build_game_rows(g.game_id, json.loads(pbp_path.read_text()), box, players, ctx, g.date)
            if g.season not in handles:
                handles[g.season] = (out_dir / f"statlines_{g.season:02d}.jsonl").open("w")
            for row in rows:
                handles[g.season].write(json.dumps(row, separators=(",", ":")) + "\n")
            counts["rows"] += len(rows)
            counts["games"] += 1
            if counts["games"] % 500 == 0:
                log(f"{counts['games']} games, {counts['rows']} rows")
        team_ledger.add(g)
        players.add(boxscore_lines(box), g.date)
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
