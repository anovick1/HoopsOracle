"""Which snapshot fields carry signal? Gradient-boosted trees on the numbers.

Trains one tree model per question on the graded logs, once per feature group
(game state only, + team context, + player context), and scores each on the
latest games. Two uses:

1. Decide what earns its tokens in the Laya snapshot: a field that does not
   help the tree is not worth sending to the encoder.
2. Set the honest bar. On pure numbers a tree is a strong baseline. Laya has
   to beat this, not just the hand-written rules.

    python finetune/ablation.py logs/season_25 [logs/season_24 ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

GROUPS = ("game", "teams", "players")
TEST_FRAC = 0.15


def load(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            for line in f:
                rows.append(json.loads(line))
    return rows


def _num(x, default=0.0):
    return float(x) if x is not None else default


def game_features(s, row):
    home, away = s["game"].split(" @ ")[1], s["game"].split(" @ ")[0]
    score = s["score"]
    margin = score[home] - score[away]
    clock = s["clock"].split(":")
    secs_period = int(clock[0]) * 60 + int(clock[1])
    period = s["period"]
    left = secs_period + max(0, 4 - period) * 720 if period <= 4 else secs_period
    run = s.get("run") or {}
    run_pts = run.get("points", 0) * (1 if run.get("team") == home else -1)
    l3 = s.get("last_3min", {})
    tol = s.get("timeouts_left", {})
    tf = s.get("team_fouls_period", {})
    bonus = s.get("in_bonus", [])
    lt = s.get("last_timeout") or {}
    return {
        "period": period, "left": left, "margin": margin, "abs_margin": abs(margin),
        "home_ball": int(s.get("possession") == home),
        "run_pts": run_pts, "run_poss": run.get("possessions", 0),
        "l3_diff": l3.get(home, 0) - l3.get(away, 0),
        "to_home": tol.get(home, 0), "to_away": tol.get(away, 0),
        "tf_home": tf.get(home, 0), "tf_away": tf.get(away, 0),
        "bonus_home": int(home in bonus), "bonus_away": int(away in bonus),
        "since_to": lt.get("seconds_ago", 9999),
        "wp_rules": row.get("win_prob_home", 0.5),
    }


def team_features(s):
    t = s.get("teams") or {}
    home, away = s["game"].split(" @ ")[1], s["game"].split(" @ ")[0]
    h, a = t.get(home, {}), t.get(away, {})
    if not h or not a:
        return {}
    out = {}
    for k in ("net", "off", "def", "pace", "three", "ft", "opp_three", "rest"):
        out[f"h_{k}"] = _num(h.get(k)); out[f"a_{k}"] = _num(a.get(k))
    out["net_diff"] = out["h_net"] - out["a_net"]
    return out


def ball_team_features(s):
    """Team context for the team with the ball (possession_scores, score_type)."""
    t = s.get("teams") or {}
    ball = s.get("possession")
    home, away = s["game"].split(" @ ")[1], s["game"].split(" @ ")[0]
    opp = away if ball == home else home
    b, o = t.get(ball, {}), t.get(opp, {})
    if not b or not o:
        return {}
    return {"b_off": _num(b.get("off")), "b_three": _num(b.get("three")), "b_ft": _num(b.get("ft")),
            "o_def": _num(o.get("def")), "o_opp_three": _num(o.get("opp_three")), "b_pace": _num(b.get("pace"))}


def label_index(q, label, criteria):
    if q["type"] == "noul":
        return int(bool(label))
    return list(criteria).index(label)


def build(rows, question):
    """X per group, y, and game ids. shooter is handled per candidate."""
    X = {g: [] for g in GROUPS}
    y, games = [], []
    for r in rows:
        if question not in r.get("labels", {}) or question not in r["questions"]:
            continue
        s = r["snapshot"]
        q = r["questions"][question]
        gf = game_features(s, r)
        tf = team_features(s) if question in ("winner", "comeback", "run_continues") else ball_team_features(s)
        if question in ("winner", "comeback", "run_continues"):
            tf = {**tf, **ball_team_features(s)}
        X["game"].append(gf)
        X["teams"].append(tf)
        X["players"].append({})
        y.append(label_index(q, r["labels"][question], q.get("criteria", {})))
        games.append((r.get("game_date") or "", r["game_id"]))
    return X, np.array(y), games


def build_shooter(rows):
    """One row per (possession, candidate). Label 1 for the actual shooter."""
    X = {g: [] for g in GROUPS}
    y, groups, games = [], [], []
    gid = 0
    for r in rows:
        if "shooter" not in r.get("labels", {}) or "shooter" not in r["questions"]:
            continue
        s = r["snapshot"]
        names = list(r["questions"]["shooter"]["criteria"])
        players = s.get("players") or {}
        gf = game_features(s, r)
        clutch = int(gf["period"] >= 4 and gf["left"] <= 300 and gf["abs_margin"] <= 8)
        tf = ball_team_features(s)
        team_pts_tonight = sum(_num(players.get(n, {}).get("pts")) for n in names) or 1.0
        team_fga_tonight = sum(_fga(players.get(n, {})) for n in names) or 1.0
        for i, n in enumerate(names):
            p = players.get(n, {})
            fga = _fga(p)
            fgm = _fgm(p)
            X["game"].append({**gf, "slot": i, "clutch": clutch, "fouls": _num((s.get("fouls") or {}).get(s["possession"], {}).get(n))})
            X["teams"].append(tf)
            X["players"].append({
                "share": _num(p.get("share"), 0.12), "clutch_share": _num(p.get("clutch"), 0.12),
                "three": _num(p.get("three"), 0.4), "pts": _num(p.get("pts")), "fga": fga, "fgm": fgm,
                "fg_pct": fgm / fga if fga else 0.4, "pts_share": _num(p.get("pts")) / team_pts_tonight,
                "fga_share": fga / team_fga_tonight, "has_profile": int("share" in p),
            })
            y.append(int(n == r["labels"]["shooter"]))
            groups.append(gid)
            games.append((r.get("game_date") or "", r["game_id"]))
        gid += 1
    return X, np.array(y), np.array(groups), games


def _fga(p):
    fg = p.get("fg")
    return float(fg.split("/")[1]) if fg else 0.0


def _fgm(p):
    fg = p.get("fg")
    return float(fg.split("/")[0]) if fg else 0.0


def to_matrix(dicts, keys):
    return np.array([[d.get(k, 0.0) for k in keys] for d in dicts], dtype=float)


def split_by_game(games):
    order = sorted(set(games))
    cut = order[int(len(order) * (1 - TEST_FRAC))]
    is_test = np.array([g >= cut for g in games])
    return ~is_test, is_test


def brier_multi(P, y):
    """Mean squared shortfall on the true class, same as nba_laya.metrics."""
    p_true = P[np.arange(len(y)), y]
    return float(np.mean((1 - p_true) ** 2)), float(np.mean(P.argmax(1) == y))


def fit_eval(X_dicts, y, train, test, keys):
    if not keys:
        return None
    X = to_matrix(X_dicts, keys)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0)
    clf.fit(X[train], y[train])
    P = clf.predict_proba(X[test])
    if P.shape[1] == 2 and len(set(y)) == 2:
        full = np.zeros((len(P), 2)); full[:, clf.classes_] = P; P = full
    return brier_multi(P, y[test])


def run_question(rows, question, log=print):
    X, y, games = build(rows, question)
    if len(y) < 200:
        log(f"{question}: only {len(y)} labels, skipped"); return
    train, test = split_by_game(games)
    base = None
    log(f"\n== {question}  (train {train.sum()}, test {test.sum()})")
    keys = []
    for group in GROUPS:
        gkeys = sorted({k for d in X[group] for k in d})
        if not gkeys:
            continue
        keys = keys + gkeys
        merged = [{**a, **b, **c} for a, b, c in zip(X["game"], X["teams"], X["players"])]
        res = fit_eval(merged, y, train, test, keys)
        brier, acc = res
        skill = "" if base is None else f"  skill vs game-only {1 - brier / base:+.1%}"
        base = base or brier
        log(f"  +{group:<8} brier {brier:.4f}  acc {acc:.3f}{skill}")
    # rules for reference on the same test rows
    rules_b = _rules_brier(rows, question, test_games={g for g, t in zip(games, test) if t})
    if rules_b is not None:
        log(f"  rules     brier {rules_b:.4f}")


def _rules_brier(rows, question, test_games):
    from nba_laya.metrics import summarize
    sel = [r for r in rows if (r.get("game_date") or "", r["game_id"]) in test_games]
    s = summarize(sel, question)
    return s["brier"] if s else None


def run_shooter(rows, log=print):
    X, y, groups, games = build_shooter(rows)
    train, test = split_by_game(games)
    log(f"\n== shooter  (candidates train {train.sum()}, test {test.sum()})")
    keys = []
    base = None
    for group in GROUPS:
        gkeys = sorted({k for d in X[group] for k in d})
        if not gkeys:
            continue
        keys = keys + gkeys
        merged = [{**a, **b, **c} for a, b, c in zip(X["game"], X["teams"], X["players"])]
        Xm = to_matrix(merged, keys)
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0)
        clf.fit(Xm[train], y[train])
        p = clf.predict_proba(Xm[test])[:, 1]
        # normalize within each possession's five candidates
        g_test = groups[test]; y_test = y[test]
        sums = defaultdict(float)
        for gi, pi in zip(g_test, p):
            sums[gi] += pi
        p_norm = np.array([pi / sums[gi] for gi, pi in zip(g_test, p)])
        # Brier on the true shooter's probability; accuracy = argmax per possession
        true_p = p_norm[y_test == 1]
        brier = float(np.mean((1 - true_p) ** 2))
        best = defaultdict(lambda: (-1.0, 0))
        for gi, pi, yi in zip(g_test, p_norm, y_test):
            if pi > best[gi][0]:
                best[gi] = (pi, yi)
        acc = float(np.mean([v[1] for v in best.values()]))
        skill = "" if base is None else f"  skill vs game-only {1 - brier / base:+.1%}"
        base = base or brier
        log(f"  +{group:<8} brier {brier:.4f}  acc {acc:.3f}{skill}")
    rules_b = _rules_brier(rows, "shooter", test_games={g for g, t in zip(games, test) if t})
    if rules_b is not None:
        log(f"  rules     brier {rules_b:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log_dirs", nargs="+", type=Path)
    ap.add_argument("--max-rows", type=int, default=0)
    args = ap.parse_args()
    paths = sorted(p for d in args.log_dirs for p in d.glob("*.jsonl"))
    rows = load(paths)
    if args.max_rows and len(rows) > args.max_rows:
        step = len(rows) // args.max_rows
        rows = rows[::step]
    print(f"{len(rows)} possessions from {len(paths)} games")
    for q in ("winner", "possession_scores", "score_type", "run_continues", "comeback"):
        run_question(rows, q)
    run_shooter(rows)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    main()
